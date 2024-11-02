import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import random
import datetime
import cv2
#import torch

import matplotlib.pyplot as plt
import time
from ultralytics import YOLO
model = YOLO("D:/DroneSimulation/droneLanding/best.pt",verbose=False)

class CustomEnv(gym.Env):
    def __init__(self, num_history=3,vehicle_name = 'SimpleFlight'):
        # Connect to AirSim
        self.vehicle_name = vehicle_name
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True,vehicle_name = self.vehicle_name)
        self.client.armDisarm(True,vehicle_name = self.vehicle_name)
        self.client.takeoffAsync(timeout_sec = 10,vehicle_name = self.vehicle_name).join()
        self.client.moveToZAsync(-5,2,vehicle_name = self.vehicle_name).join()
        
        self.step_length = 0.5
        self.pred_altitude = 10
        # Number of historical frames to store
        self.num_history = num_history
        self.history = None
        self.consecutive_no_detection = 0  # Track number of timesteps without platform detection
        self.step_count = 0
        self.max_steps = 100
        self.total_reward = 0
        self.platform_detected = False
        # Define the action and observation space
        self.action_space = spaces.Tuple((
                                        spaces.Discrete(7),  # Integer from 0 to 7
                                        spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype=np.float32)  # Float value
                                    ))
        # tracker ID + bounding box info (x_min, y_min, x_max, y_max, center_x, center_y, area)
        self.observation_space = spaces.Dict({
                                                "box_info": spaces.Box(low=0, high=1, shape=(self.num_history,8), dtype=np.float32),  # 7 + 1 for tracker ID
                                                "image": spaces.Box(low=0, high=255, shape=(255, 256, 1), dtype=np.uint8)
                                            })
        self.drone_pose = self.client.simGetVehiclePose(vehicle_name = self.vehicle_name)
        # Load YOLOv8 model with tracking capability
        self.model = model
        
    def reset(self, seed=None, options=None):
        # Reset the drone and the environment
        #self.client.reset()
        #self.client.takeoffAsync(timeout_sec = 10,vehicle_name = self.vehicle_name).join()
        #self.client.moveToZAsync(-8,2,vehicle_name = self.vehicle_name).join()
        self.client.simSetVehiclePose(pose=self.drone_pose, ignore_collision=True,vehicle_name = self.vehicle_name)
        self.total_reward = 0
        self.platform_detected = False
        self.pred_altitude = 10
        # Reset bounding box history and step_count
        self.history = None

        self.step_count = 0

        infos = {}
        return self._get_bounding_box_info(), infos

    def step(self, action):
        # Apply the action to the drone
        '''
        vx = float(action[0])
        vy = float(action[1])
        vz = float(action[2])
        
        print('vx: ',vx)
        print('vy: ',vy)
        print('vz: ',vz)
        
        self.client.moveByVelocityBodyFrameAsync(vx, vy, vz, 10)
        '''
        self.pred_altitude = action[1]
        #print(action)
        quad_offset = self.interpret_action(action[0])
        #print(quad_offset)
        quad_vel = self.client.getMultirotorState(vehicle_name = self.vehicle_name).kinematics_estimated.linear_velocity
        self.client.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            10,vehicle_name = self.vehicle_name       
        )

        # Get new bounding box information and update the history
        self.history = self._get_bounding_box_info()
        
        # Calculate the reward, terminations, and truncations
        reward, terminations, truncations = self._compute_reward()

        
        return self.history, reward, terminations, truncations, {}

    def render(self, mode="human", close=False):
        pass


    #--------------Addition function----------------
    def interpret_action(self, action):
        
        if action == 1:
            quad_offset = (self.step_length, 0, 0)
        elif action == 2:
            quad_offset = (0, self.step_length, 0)
        elif action == 3:
            quad_offset = (0, 0, self.step_length)
        elif action == 4:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 5:
            quad_offset = (0, -self.step_length, 0)
        elif action == 6:
            quad_offset = (0, 0, -self.step_length)
        else:
            quad_offset = (0, 0, 0)

        return quad_offset
    
    def _get_bounding_box_info(self):
        obs = {
                    "box_info": None,
                    "image": None
                }
        # Get the camera image from AirSim
        try:
            response = self.client.simGetImages([airsim.ImageRequest("3", airsim.ImageType.Scene, False, False)],vehicle_name = self.vehicle_name)
            img1d = np.frombuffer(response[0].image_data_uint8, dtype=np.uint8)

            # If image data is empty, generate a blank image
            if img1d.size == 0:
                raise ValueError("No image data received")
                
            # Reshape the image data into RGB format
            img_rgb = img1d.reshape(response[0].height, response[0].width, 3)
        except Exception as e:
            # If there's an error receiving the image, generate a blank placeholder image
            print(f"Error receiving image: {e}. Using a blank placeholder image.")
            img_rgb = np.zeros((256, 256, 3), dtype=np.uint8)

        # Run detection on the image
        results = self.model.track(img_rgb,show= True, conf = 0.5,verbose=False)  # Perform detection

        # Check if a landing pad is detected
        if len(results[0].boxes) > 0:
            # Get the first detection's bounding box coordinates
            box = results[0].boxes[0]  # Get the first detected box
            x_min, y_min, x_max, y_max = box.xyxy[0]  # Extract bounding box coordinates
            tracker_id = np.float32(box.id)
            
            # Calculate the center of the bounding box
            center_x = (x_min + x_max) / 2 / img_rgb.shape[1]  # Normalize to image width
            center_y = (y_min + y_max) / 2 / img_rgb.shape[0]  # Normalize to image height

            # Calculate the area of the bounding box (relative to image size)
            box_width = (x_max - x_min) / img_rgb.shape[1]
            box_height = (y_max - y_min) / img_rgb.shape[0]
            box_area = box_width * box_height

            # Create the bounding box info (normalized values)
            box_info = np.array([[1,  
                                 x_min / img_rgb.shape[1],
                                 y_min / img_rgb.shape[0],
                                 x_max / img_rgb.shape[1],
                                 y_max / img_rgb.shape[0],
                                 center_x,
                                 center_y,
                                 box_area]], dtype=np.float32)
            self.platform_detected = True
            self.consecutive_no_detection = 0  # Reset no-detection counter
        else:
            # If no landing pad detected, use default values and increment the no-detection counter
            box_info = np.array([[0, 0, 0, 0, 0, -1, -1, 0]], dtype=np.float32)  # Default values (center and no area)
            self.platform_detected = False
            self.consecutive_no_detection += 1

        obs["box_info"] = box_info
        obs["image"] = img_rgb
        return obs

    def _compute_reward(self):
        # Initialize reward for this timestep
        reward = 0
        
        # Check for collision first
        if self.client.simGetCollisionInfo(vehicle_name = self.vehicle_name).has_collided:
            # Set values for this agent and continue to the next agent
            #print("Collision detected")
            reward = -100
            terminations = True
            truncations = False
            return reward, terminations, truncations
        

        
        # Use the most recent bounding box info
        current_box = self.history[-1]
        cX, cY = current_box[5], current_box[6]  # Get center of the latest bounding box (relative coordinates)
        box_area = current_box[7]  # Get the bounding box area
        detected_pad = self.platform_detected  # Determine if the pad is detected (tracker ID not 0)

        # Detection reward
        if detected_pad:
            reward += 1  # +1 if the pad is detected
        else:
            reward -= 2  # -1 if the pad is not detected

        # If the pad is detected, calculate alignment reward
        if detected_pad:
            # Calculate the distance from the image center to the bounding box center
            distance_from_center = np.sqrt((cX - 0.5) ** 2 + (cY - 0.5) ** 2)

            # Alignment reward based on distance
            if distance_from_center < 0.2:
                reward += 1
            elif 0.2 <= distance_from_center < 0.5:
                reward += 0.1
            else:
                reward -= 1

            # Apply additional reward based on bounding box size (indicating closeness)
            reward += min(box_area * 5, 1.0)  # As the drone gets closer (larger box), give more reward
        
        
        # Altitude-based scaling
        position = self.client.getMultirotorState(vehicle_name = self.vehicle_name).kinematics_estimated.position
        velocity = self.client.getMultirotorState(vehicle_name = self.vehicle_name).kinematics_estimated.linear_velocity
        velocity_magnitude = np.linalg.norm([velocity.x_val, velocity.y_val, velocity.z_val])
        altitude = -position.z_val  # AirSim uses right-handed coordinates, so altitude is -z

        margin_of_error = 0.1 * altitude
        if abs(self.pred_altitude - altitude) <= margin_of_error:
            reward += 1  # Within the margin
        else:
            reward += -1  # Outside the margin
            
        # Multiply the reward by altitude scaling
        reward += max(0, 1 - altitude / 10)
        

        # Landing reward check
        is_centered = detected_pad and distance_from_center < 0.2  # Check if centered when landing
        is_low_altitude = altitude < 0.3  # Consider it landed if altitude is very low
        is_low_velocity = velocity_magnitude < 0.5  # Low velocity landing

        if is_low_altitude and is_low_velocity:
            terminations = True
            if is_centered:
                reward += 20  # Large reward for successful, centered landing
            else:
                reward -= 5  # Penalty for off-center landing
        
        # Time penalty to encourage faster completion
        reward -= 3
        reward = round(reward, 2)
        # Update the total reward
        self.total_reward += reward
        #print(self.total_reward, end=" ")
 
        # Termination condition if the total reward gets too low
        terminations = False
        if self.total_reward < -20:
            terminations = True

        # Update the step count
        self.step_count += 1
        truncations = self.step_count >= self.max_steps

        # Apply truncation penalty if episode takes too long
        if truncations:
            reward -= 10

        # Check for loss of helipad visibility for 8 consecutive frames
        if self.consecutive_no_detection >= 10:
            terminations = True
            reward -= 10  # Penalty for losing sight of the platform

        return reward, terminations, truncations


