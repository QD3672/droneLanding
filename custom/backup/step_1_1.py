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
        self.drone_process = self.multirotor_client.takeoffAsync(timeout_sec = 1, vehicle_name = agent_name)
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        #self.midas_transform = self.get_midas_transform()
        
        self.max_steps = 100
        self.agent_name=agent_name
        # Action space
        self.action_space = spaces.Discrete(7)

        # Observation space
        self.image_space = spaces.Box(low=0, high=255, shape=(144, 256, 3), dtype=np.uint8)
        self.orientation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.location_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self.observation_space =  spaces.Dict({
                                                'image': self.image_space,
                                                'location': self.location_space,
                                                'target': self.location_space,
                                                'collision': spaces.MultiBinary(1),
                                            })
   
        #self.multirotor_client = __multirotor_client
        
        self.pose = self.multirotor_client.simGetVehiclePose(vehicle_name = self.agent_name)
        self.kinematics_estimated = self.multirotor_client.getMultirotorState(vehicle_name = self.agent_name).kinematics_estimated
        
        # Agent attribute
        self.step_length = 0.5
        self.agent_current_step = 0
        self.agent_old_locations = self.__get_multirotor_location()
        self.agent_current_location = None
        self.agent_vel = self.kinematics_estimated.linear_velocity
        
        self.corrupt = False
        self.accumulate_reward = 0
        self.observations = {
                            'image': None,
                            'location': None,
                            'target': None,
                            'collision': np.array([0]),
                        }
        
        self.rewards = 0
        self.terminations = False
        self.truncations = False
        self.last_action = None
        
        #Init target list
        self.__target_point_list = []
        self.__init_target_point_list()
        self.__target_point = random.choice(self.__target_point_list).astype(np.float32)
        self.pts = []

        # Init agent attribute
        self.get_obs()
        self.start_time = time.time()

    def reset(self, seed=None, options=None):
        self.agent_current_step = 0 
        self.rewards = 0
        self.accumulate_reward = 0
        self.terminations = False
        self.truncations = False
        self.__target_point = random.choice(self.__target_point_list).astype(np.float32)

        self.agent_vel.x_val = 0
        self.agent_vel.y_val = 0
        self.agent_vel.z_val = 0
        
        #self.__set_agent_location(self.__get_new_start_locations())
        #self.multirotor_client.hoverAsync(vehicle_name = self.agent_name).join()
        #self.multirotor_client.moveByVelocityAsync(0.01, -0.01, -0.01, 0.5,vehicle_name = self.agent_name).join()
                
        #self.agent_old_locations = self.__get_multirotor_location()

        #self.pts = self.generate_trajectory_with_min_height(self.agent_old_locations,self.__target_point)

        #self.observations["collision"] = np.array([0])
        # Get the current observations
        #self.get_obs()
        
        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {}
        #print(self.__target_point)
        return self.observations, infos

    def step(self, action):
        
        
        #print(self.agent_name," : ",time.time()-self.start_time)
        
        self._do_action(action)
        #self.multirotor_client.simPause(True)
        self.kinematics_estimated = self.multirotor_client.getMultirotorState(vehicle_name = self.agent_name).kinematics_estimated
        #self.agent_vel = self.kinematics_estimated.linear_velocity
        
        # Get observations
        self.get_obs()
        
        # Compute reward
        self.compute_reward()
        
        #self.multirotor_client.simPause(False)
        
        self.agent_old_locations =  self.agent_current_location.copy()
        
        # Increment step counter for each agent
        self.agent_current_step += 1
        
        infos = {}
        #self.start_time = time.time()
        return self.observations, self.rewards, self.terminations, self.truncations, infos

    def render(self, mode="human", close=False):
        pass


    #--------------Addition function----------------
    def get_midas_transform(self,model_type = "MiDaS_small"):
        #model_type = "DPT_Large"# MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.midas.to(self.device)
        self.midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            return midas_transforms.dpt_transform
        else:
            return midas_transforms.small_transform

    def generate_trajectory_with_min_height(self, A, B, num_points = 5, min_height = -100):
        """
        Generate a trajectory between two points A and B using linear interpolation with a minimum height.

        Parameters:
        - A (tuple): Starting point (x, y, z)
        - B (tuple): Ending point (x, y, z)
        - num_points (int): Number of points in the trajectory
        - min_height (float): Minimum height for the trajectory

        Returns:
        - trajectory (list of tuples): List of points representing the trajectory
        """
        x_A, y_A, z_A = A
        x_B, y_B, z_B = B

        # Generate linearly spaced values for x and y
        x_trajectory = np.linspace(x_A, x_B, num_points)
        y_trajectory = np.linspace(y_A, y_B, num_points)

        # Ensure a minimum height for the z-coordinate
        z_trajectory = np.maximum(np.linspace(z_A, z_B, num_points), min_height)

        # Combine the x, y, and z values into a list of tuples representing the trajectory
        trajectory = [(x, y, z) for x, y, z in zip(x_trajectory, y_trajectory, z_trajectory)]

        return trajectory
    
    def get_obs(self):
        self.observations["image"] = self.__get_image()
        self.observations["location"] = self.__get_multirotor_location()
        self.observations["target"] = self.__target_point
    
    # Gets an image from AirSim
    def __get_image(self,image_type = "rgb"):
        image_response = self.multirotor_client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthVis, False, False)],vehicle_name = self.agent_name)[0]
        image1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
        image_rgba = image1d.reshape(image_response.height, image_response.width, 3)

        # Check if the reshaped image has the expected shape
        if image_rgba.shape != (144, 256, 3):
            # If not, create a new image of the correct size
            image_rgba = np.zeros((144, 256, 3), dtype=np.uint8)

        image_rgba = image_rgba[:,:,0:3].astype(np.uint8)
        
        # Convert values from range [0, 255] to [0, 1]
        img = image_rgba
        return img
        '''
        if image_type == "depth":
            input_batch = self.midas_transform(img).to(self.device)
            with torch.no_grad():
                prediction = self.midas(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                output = prediction.cpu().numpy()
                formatted = (output * 255 / np.max(output)).astype('uint8')
                #plt.imshow(formatted)
                #plt.title('Your Image')
                #plt.show()
            return formatted
            
        else:'''
            

    def __get_multirotor_location(self):
        position = self.kinematics_estimated.position
        self.agent_current_location = np.array([position.x_val, position.y_val, position.z_val]).astype(np.float32)
        return self.agent_current_location

    def __get_multirotor_orientation(self):
        orientation = self.multirotor_client.simGetVehiclePose(vehicle_name = self.agent_name).orientation
        return np.array([orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val]).astype(np.float32)
        
    def __get_new_start_locations(self):
        random_choice = np.random.choice([1, 2, 3])
        if random_choice == 1:
            return np.random.uniform(low=(-40, -10,-2), high=(-25, 10,-2), size=(3,))
        elif random_choice == 2:
            return np.random.uniform(low=(35, -100,-2), high=(50, -80,-2), size=(3,))
        else:
            return np.random.uniform(low=(145, -100,-2), high=(155, -50,-2), size=(3,))

    def __set_agent_location(self, locations):
        pose = self.multirotor_client.simGetVehiclePose(vehicle_name = self.agent_name)           
        pose.position.x_val = locations[0]
        pose.position.y_val = locations[1]
        pose.position.z_val = locations[2]
        
        self.multirotor_client.simSetVehiclePose(pose=pose, ignore_collision=True,vehicle_name = self.agent_name)

    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        self.agent_vel.x_val = self.agent_vel.x_val * 0.7 + quad_offset[0]
        self.agent_vel.y_val = self.agent_vel.y_val * 0.7 + quad_offset[1]
        self.agent_vel.z_val = self.agent_vel.z_val * 0.7 + quad_offset[2]
           
        #self.drone_process =
        self.multirotor_client.moveByVelocityAsync(
                                self.agent_vel.x_val,
                                self.agent_vel.y_val,
                                self.agent_vel.z_val,
                                2,
                                drivetrain =airsim.DrivetrainType.ForwardOnly,
                                yaw_mode = airsim.YawMode(False,0),
                                vehicle_name = self.agent_name,
                            )

    def interpret_action(self, action):
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            quad_offset = (0, 0, self.step_length)
        elif action == 3:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 4:
            quad_offset = (0, -self.step_length, 0)
        elif action == 5:
            quad_offset = (0, 0, -self.step_length)
        else:
            quad_offset = (0, 0, 0)

        return quad_offset
    
    def __init_target_point_list(self):
        with open('target_points.txt', 'r') as f:
            for line in f:
                target_location = np.array(line.split(','))
                self.__target_point_list.append(target_location)


    def __is_x_direction_pointing_to_target(self, agent_location,
                                          agent_orientation,
                                          target_location,
                                          tolerance=0.1):
        # Convert quaternion to rotation matrix using pyquaternion
        rotation_matrix_A = Quaternion(agent_orientation).rotation_matrix

        # Extract x-direction vector from the rotation matrix
        x_direction_A = rotation_matrix_A[:, 0]

        # Vector from A to B
        vector_AB = target_location - agent_location

        # Check if the dot product is within the tolerance range
        dot_product = np.dot(x_direction_A, vector_AB)
        tolerance_range = np.linalg.norm(x_direction_A) * np.linalg.norm(vector_AB) * tolerance

        # If dot product is within the tolerance range, x-direction of A is pointing towards B
        is_pointing_to_target = dot_product > tolerance_range
        # Calculate the angle in degrees using arccosine
        angle_in_radians = np.arccos(dot_product / (np.linalg.norm(x_direction_A) * np.linalg.norm(vector_AB)))
        # Convert radians to degrees
        angle_in_degrees = np.degrees(angle_in_radians)
        
        return is_pointing_to_target, angle_in_degrees

    def calculate_rotation_quaternion(self,location_A, location_B):
        # Normalize the vectors to ensure they represent orientations
        vector_A = location_A / np.linalg.norm(location_A)
        vector_B = location_B / np.linalg.norm(location_B)
        print(vector_A)
        print(vector_B)
        # Calculate the rotation axis and angle between the vectors
        rotation_axis = np.cross(vector_A, vector_B)
        rotation_angle = np.arccos(np.dot(vector_A, vector_B))

        # Create the quaternion
        quaternion = Quaternion(axis=rotation_axis, angle=rotation_angle)

        return quaternion

    def quaternion_to_direction(self,q):
        """
        Convert a quaternion to a direction vector.

        Parameters:
        q (numpy.array): A quaternion represented as a numpy array [w, x, y, z]

        Returns:
        numpy.array: A direction vector derived from the quaternion
        """
        # Normalize the quaternion if it's not already normalized
        q = q / np.linalg.norm(q)

        # Convert the quaternion into a direction vector
        direction = q[1:]  # exclude the w-component of the quaternion

        # Normalize the direction vector
        direction = direction / np.linalg.norm(direction)

        return direction

    def are_vectors_in_same_direction(self,vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        
        if dot_product > 0:
            return True  # Vectors are in the same direction
        elif dot_product < 0:
            return False  # Vectors are in opposite directions
        else:
            return True  # Vectors are collinear (may have different magnitudes)
                
    def compute_reward(self):
        THRESH_DIST = 7
        SPEED_REDUCTION = 1
        BETA = 1
        self.rewards = 0 
        
        # Check for collision first
        if self.multirotor_client.simGetCollisionInfo(vehicle_name = self.agent_name).has_collided:
            # Set values for this agent and continue to the next agent
            #print("Collision detected")
            self.rewards += -10
            self.observations["collision"] = np.array([1])
            #self.terminations = True
        else:
            self.observations["collision"] = np.array([0])

        #multirotor_state = self.multirotor_client.getMultirotorState(vehicle_name = self.agent_name)
        #orientation = multirotor_state.kinematics_estimated.orientation
        '''
        quad_pt = self.agent_current_location

        quad_vel_orgin = self.agent_vel
        quad_vel = [quad_vel_orgin.x_val, quad_vel_orgin.y_val, quad_vel_orgin.z_val]
        pts = self.pts
        dist = np.min(
            np.linalg.norm(
                np.cross((quad_pt - pts[:-1]), (quad_pt - np.roll(pts, -1, axis=0)[:-1])),
                axis=1,
            )
            / np.linalg.norm(np.diff(pts, axis=0), axis=1)
        )
        if dist > THRESH_DIST:
            self.rewards += -5
        else:
            reward_dist = math.exp(-BETA * dist) - 0.5
            reward_speed = np.linalg.norm(quad_vel) - SPEED_REDUCTION
            self.rewards += (reward_dist + reward_speed)                
  
        
        print()
        print("old_distance ",old_distance)
        print("new_distance ",new_distance)
        print()
        '''
        q = self.kinematics_estimated.orientation.to_numpy_array()
        d = self.quaternion_to_direction(q)
        vector1= d
        vector2=self.__target_point-self.agent_current_location

        if self.are_vectors_in_same_direction(vector1, vector2):
            self.rewards += 1
        else:
            self.rewards += -1 
            
        old_distance = np.linalg.norm(self.agent_old_locations - self.__target_point)
        new_distance = np.linalg.norm(self.agent_current_location - self.__target_point)
          
        # Initialize reward and status
        if new_distance < old_distance:
            self.rewards += 1
        else:
            self.rewards += -1 

        #reward += 1 if orientation.x_val < 0.1 or orientation.y_val < 0.1 else 0
        if new_distance < 2:
            self.rewards += 100
            print("target aquire")
        self.terminations = new_distance < 2
        
        # Check truncation condition
        truncated = self.agent_current_step >= self.max_steps
        self.truncations = truncated
        self.accumulate_reward += self.rewards
        if self.accumulate_reward < -11:
            self.terminations = True
