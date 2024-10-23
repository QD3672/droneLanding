import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import random
import datetime
from pyquaternion import Quaternion
import cv2
import torch
import math
import matplotlib.pyplot as plt
import time

class CustomEnv(gym.Env):
    def __init__(self, agent_name='SimpleFlight',__multirotor_client=None):
        self.multirotor_client = airsim.MultirotorClient()
        self.multirotor_client.confirmConnection()
        self.multirotor_client.enableApiControl(True,vehicle_name = agent_name)
        self.multirotor_client.armDisarm(True,vehicle_name = agent_name)
        self.drone_process = self.multirotor_client.takeoffAsync(timeout_sec = 10, vehicle_name = agent_name)
        self.agent_name=agent_name
        
        # Action space
        self.action_space = spaces.Discrete(3)

        # Observation space
        self.observation_space =  spaces.Box(low=0, high=255, shape=(154, 256, 1), dtype=np.uint8)

        # flight height
        self.z = -10
        self.max_steps = 1000

        # Drone attribute
        self.kinematics_estimated = self.multirotor_client.getMultirotorState(vehicle_name = self.agent_name).kinematics_estimated
        self.agent_current_location = self.kinematics_estimated.position.to_numpy_array()

        #Init target list
        self.__target_point_list = []
        self.__init_target_point_list()
        self.__target_point = random.choice(self.__target_point_list).astype(np.float32)
        
    def reset(self, seed=None, options=None):
        self.agent_current_step = 0
        self.rewards = 0
        self.accumulate_reward = 0
        self.terminations = False
        self.truncations = False
        self.__target_point = random.choice(self.__target_point_list).astype(np.float32)

        self.__set_agent_location(self.__get_new_start_locations())
        self.multirotor_client.moveToZAsync(self.z, 3,vehicle_name = self.agent_name)
        self.kinematics_estimated = self.multirotor_client.getMultirotorState(vehicle_name = self.agent_name).kinematics_estimated

        self.agent_old_locations = self.__get_multirotor_location()
        track=self.goal_direction(self.__target_point,self.kinematics_estimated.position.to_numpy_array())
        self.observations = self.__get_image(track)
 
        infos = {}
        return self.observations, infos

    def step(self, action):
        n_try = 0
        while self.kinematics_estimated.position.z_val < -20:
            self.multirotor_client.moveToZAsync(self.z, 3,vehicle_name = self.agent_name).join()
            self.kinematics_estimated = self.multirotor_client.getMultirotorState(vehicle_name = self.agent_name).kinematics_estimated
            print("Current height: ",self.kinematics_estimated.position.z_val)
            n_try +=1
            if n_try > 10:
                self.truncations = True
                break

        if action == 0:
            self.move_straight(1,3)
        elif action == 1:
            self.rotate_clockwise(1)
        else:
            self.rotate_counterClockwise(1)
            
        #self.stop_drone()
        self.kinematics_estimated = self.multirotor_client.getMultirotorState(vehicle_name = self.agent_name).kinematics_estimated
        self.__get_multirotor_location()
        self.compute_reward()
        track=self.goal_direction(self.__target_point,self.kinematics_estimated.position.to_numpy_array())
        self.observations = self.__get_image(track)
        self.agent_old_locations =  self.agent_current_location.copy()
        infos = {}
 
        return self.observations, self.rewards, self.terminations, self.truncations, infos

    def render(self, mode="human", close=False):
        pass


    #--------------Addition function----------------
    def __init_target_point_list(self):
        self.__target_point_list = [np.array([150,80,-10]),
                                    np.array([150,-90,-10])]
        
    def __set_agent_location(self, locations):
        pose = self.multirotor_client.simGetVehiclePose(vehicle_name = self.agent_name)           
        pose.position.x_val = locations[0]
        pose.position.y_val = locations[1]
        pose.position.z_val = locations[2]
        
        self.multirotor_client.simSetVehiclePose(pose=pose, ignore_collision=True,vehicle_name = self.agent_name)

    def __get_new_start_locations(self):
        return np.random.uniform(low=(-10, -122,-2), high=(-10, 122,-2), size=(3,))
        
    def __get_multirotor_location(self):
        self.agent_current_location = self.kinematics_estimated.position.to_numpy_array()
        return self.agent_current_location
    
    def __get_image(self,track):
        response = self.multirotor_client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthVis, True)],vehicle_name = self.agent_name)[0]
        image1d = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
        image_gray = np.expand_dims(image1d, axis=-1)
        image_gray = image_gray.copy()
        image_gray[image_gray > 1] = 1
        image_gray = np.array(image_gray * 255, dtype=np.uint8)
        if image_gray.shape != (144,256,1):
            image_gray = np.zeros((144,256,1), dtype=np.uint8)
        cut = image_gray[20:40,:,]
        info_section = np.zeros((10,cut.shape[1],cut.shape[2]),dtype=np.uint8) + 255
        info_section[9,:,:] = 0
        
        line = int((((track - -180) * (100 - 0)) / (180 - -180)) + 0)
        
        if line != (0 or 100):
            info_section[:,line-1:line+2,:]  = 0
        elif line == 0:
            info_section[:,0:3,:]  = 0
        elif line == 100:
            info_section[:,info_section.shape[1]-3:info_section.shape[1],:]  = 0
            
        total = np.concatenate((info_section, image_gray), axis=0)
        
        return total
    
    def move_straight(self,duration,speed):
        pitch, roll, yaw  = airsim.to_eularian_angles(self.kinematics_estimated.orientation)
        vx = np.cos(yaw) * speed
        vy = np.sin(yaw) * speed
        self.multirotor_client.moveByVelocityZAsync(vx,vy,self.z,duration,drivetrain =airsim.DrivetrainType.ForwardOnly,vehicle_name =self.agent_name).join()

    def rotate_clockwise(self,duration):
        self.multirotor_client.rotateByYawRateAsync(30,duration,vehicle_name = self.agent_name).join()

    def rotate_counterClockwise(self,duration):
        self.multirotor_client.rotateByYawRateAsync(-30,duration,vehicle_name = self.agent_name).join()

    def stop_drone(self):
        self.multirotor_client.moveByVelocityAsync(
                                0,0,0,1,
                                vehicle_name = self.agent_name,
                            ).join()
        self.multirotor_client.rotateByYawRateAsync(0,1,vehicle_name = self.agent_name).join()

    def compute_reward(self):
        if self.multirotor_client.simGetCollisionInfo(vehicle_name = self.agent_name).has_collided:
            # Set values for this agent and continue to the next agent
            #print("Collision detected")
            self.rewards += -10
            self.terminations = True
            
        old_distance = np.linalg.norm(self.agent_old_locations - self.__target_point)
        new_distance = np.linalg.norm(self.agent_current_location - self.__target_point)
    
        # Initialize reward and status
        
        self.rewards += -1 + old_distance - new_distance

        if new_distance < 3:
            self.rewards += 100
            print("target aquire")

    def goal_direction(self, goal,pos):
        pitch, roll, yaw  = airsim.to_eularian_angles(self.multirotor_client.getMultirotorState(vehicle_name = self.agent_name).kinematics_estimated.orientation)
        yaw = np.degrees(yaw)
       
        pos_angle = np.arctan2(goal[1] - pos[1], goal[0] - pos[0])
        pos_angle = np.degrees(pos_angle) % 360
       
        track = np.radians(pos_angle - yaw)
       
        return ((np.degrees(track) - 180) % 360) - 180
