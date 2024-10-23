import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import random
import datetime
from pyquaternion import Quaternion
import cv2
#import torch

import matplotlib.pyplot as plt
import time

class CustomEnv(gym.Env):
    def __init__(self, agent_name='SimpleFlight',__multirotor_client=None):
        self.multirotor_client = airsim.MultirotorClient()
        self.multirotor_client.confirmConnection()
        self.multirotor_client.enableApiControl(True,vehicle_name = agent_name)
        self.multirotor_client.armDisarm(True,vehicle_name = agent_name)
        self.drone_process = self.multirotor_client.takeoffAsync(timeout_sec = 10, vehicle_name = agent_name)


        #self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        #self.midas_transform = self.get_midas_transform()
        self.z = -5
        self.max_steps = 10000 
        self.agent_name=agent_name
        # Action space
        self.action_space = spaces.Discrete(3)

        # Observation space
        self.image_space = spaces.Box(low=0, high=255, shape=(144, 256, 1), dtype=np.uint8)
        self.velocity_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.distance_sensor_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.orientation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.location_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.distance_to_target = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        
        self.observation_space =  spaces.Dict({
                                                'image': self.image_space,
                                                'distance_to_target': self.distance_to_target
                                            })
   
        #self.multirotor_client = __multirotor_client
        
        self.pose = self.multirotor_client.simGetVehiclePose(vehicle_name = self.agent_name)
        self.kinematics_estimated = self.multirotor_client.getMultirotorState(vehicle_name = self.agent_name).kinematics_estimated
        
        # Agent attribute
        self.step_length = 1
        self.agent_current_step = 0
        self.agent_old_locations = self.__get_multirotor_location()
        self.agent_current_location = None
        self.agent_vel = self.kinematics_estimated.linear_velocity
        self.current_distance_to_target = np.inf
        self.old_dist = np.inf
        
        self.corrupt = False
        self.accumulate_reward = 0
        self.observations = {
                            'image': None,
                            'distance_to_target': np.inf,
                        }
        
        self.rewards = 0
        self.terminations = False
        self.truncations = False
        self.last_action = None
        self.start_to_end_dist = 0.0
        self.total_finish_eps = 0.0
        self.total_trajectory = -1
        self.total_collision = -1
        
        #Init target list
        self.__target_point_list = []
        self.__init_target_point_list()
        self.__target_point = random.choice(self.__target_point_list).astype(np.float32)
        self.pts = []

        # Init agent attribute
        self.get_obs()
        self.start_time = time.time()

    def reset(self, seed=None, options=None):
        
        self.total_trajectory +=1
        print("total_trajectory ", self.total_trajectory)
        self.agent_current_step = 0 
        self.rewards = 0
        self.accumulate_reward = 0
        self.terminations = False
        self.truncations = False
        self.current_distance_to_target = np.inf
        self.old_dist = np.inf
        self.__set_agent_location_and_target_point()
        #self.__target_point = random.choice(self.__target_point_list).astype(np.float32)
        
        #self.observations["target"] = self.__target_point
        
        #self.multirotor_client.simPause(False)
        #self.__set_agent_location(self.__get_new_start_locations())
        self.multirotor_client.moveToZAsync(self.z, 1,vehicle_name = self.agent_name).join()
        self.multirotor_client.hoverAsync(vehicle_name = self.agent_name).join()
        #self.multirotor_client.takeoffAsync(timeout_sec = 2, vehicle_name = self.agent_name).join()
        #self.multirotor_client.simPause(True)
        
        #self.kinematics_estimated = self.multirotor_client.getMultirotorState(vehicle_name = self.agent_name).kinematics_estimated
        #self.pts = self.generate_trajectory_with_min_height(self.agent_old_locations,self.__target_point)

        # Get the current observations
        self.get_obs()
        #self.agent_old_locations = self.__get_multirotor_location()
        #self.agent_old_locations =  self.agent_current_location.copy()
        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {}
        #self.start_to_end_dist = np.linalg.norm(self.agent_current_location - self.__target_point)
        #print(self.__target_point)
        return self.observations, infos

    def step(self, action):
        #self.agent_old_locations =  self.agent_current_location.copy()
        #self.drone_process.join()
        #print(self.agent_name," : ",time.time()-self.start_time)
        #self.start_time = time.time()
        #self.multirotor_client.simPause(False)
        self._do_action(action)
        #time.sleep(0.9)
        #self.multirotor_client.simPause(True)
        #self.kinematics_estimated = self.multirotor_client.getMultirotorState(vehicle_name = self.agent_name).kinematics_estimated
        #self.agent_vel = self.kinematics_estimated.linear_velocity
        
        # Get observations
        self.get_obs()
        
        # Compute reward
        self.compute_reward()
        
        #self.agent_old_locations =  self.agent_current_location.copy()
        
        # Increment step counter for each agent
        self.agent_current_step += 1
        
        infos = {}
        #self.drone_process.join()
        return self.observations, self.rewards, self.terminations, self.truncations, infos

    def render(self, mode="human", close=False):
        pass


    #--------------Addition function----------------
    def get_midas_transform_back_up(self,model_type = "MiDaS_small"):
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

    def get_midas_transform(self):
        self.image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", low_cpu_mem_usage=True)

    def generate_trajectory_with_min_height(self, A, B, num_points = 5, min_height = -10):
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
        z_trajectory[0] = z_A
        z_trajectory[-1] = z_B
        # Combine the x, y, and z values into a list of tuples representing the trajectory
        trajectory = [(x, y, z) for x, y, z in zip(x_trajectory, y_trajectory, z_trajectory)]

        return trajectory
    
    def get_obs(self):
        self.observations["image"] = self.__get_image()
        #self.observations["distance_sensor"] = [f_ds_dt.distance,d_ds_dt.distance,u_ds_dt.distance]
        #self.observations["location"] = self.__get_multirotor_location()
        #self.observations["target"] = self.__target_point
        self.observations["distance_to_target"] = np.array([self._calculate_distance_to_target()])
    
    # Gets an image from AirSim
    def __get_image(self):
        MIN_DEPTH_METERS = 0
        MAX_DEPTH_METERS = 20
        response = self.multirotor_client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthPerspective, True,False)],vehicle_name = self.agent_name)[0]
        depth_image1d = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
        #depth_image_in_meters =  depth_image1d.reshape(response.width, response.height,1)
        image_gray = np.expand_dims(depth_image1d, axis=-1)
        #image_gray = image_gray.copy()
        
        #image_gray[image_gray > 1] = 1
        #image_gray = np.array(image_gray * 255, dtype=np.uint8)
        image_gray = np.interp(image_gray, (MIN_DEPTH_METERS, MAX_DEPTH_METERS), (0, 255)).astype('uint8')
        #cv2.imwrite("depth_visualization.png", depth_8bit_lerped.astype('uint8'))
        #print(image_gray.dtype)
        """
        image = image_gray.astype(float) / 255.0
        mean = np.mean(image)
        std_dev = np.std(image)
        adj_image = (image - mean)*10*std_dev +mean

        image_gray = np.clip(adj_image,0,1)

        image_gray = np.array(image_gray * 255, dtype=np.uint8)
        """
        if image_gray.shape != (144,256,1):
            image_gray = np.zeros((144,256,1), dtype=np.uint8)
        #print(image_gray)
        #cv2.imshow("",image_gray)
        #plt.imshow(image_gray)
        #plt.show()
        return image_gray
            

    def __get_multirotor_location(self):
        self.agent_current_location = self.kinematics_estimated.position.to_numpy_array()
        return self.agent_current_location

    def __get_multirotor_orientation(self):
        orientation = self.multirotor_client.simGetVehiclePose(vehicle_name = self.agent_name).orientation.to_numpy_array()
        return orientation
        
    def __get_new_start_locations(self):
        random_choice = np.random.choice([1, 2, 3])
        if self.agent_name == 1:
            return np.array([-74,-1,-5])
        elif self.agent_name ==2 :
            return np.array([-74,16,-5])
        elif self.agent_name ==3 :
            return np.array([-74,32,-5])
        elif self.agent_name ==4 :
            return np.array([-74,48,-5])
        elif self.agent_name ==5 :
            return np.array([-74,64,-5])
        else:
            return np.array([-74,80,-5])

    def __set_agent_location(self, locations):
        pose = self.multirotor_client.simGetVehiclePose(vehicle_name = self.agent_name)           
        pose.position.x_val = locations[0]
        pose.position.y_val = locations[1]
        pose.position.z_val = locations[2]
        
        self.multirotor_client.simSetVehiclePose(pose=pose, ignore_collision=True,vehicle_name = self.agent_name)
    def __set_pose(self, pose, locations,orientation):
            pose.position.x_val = locations[0]
            pose.position.y_val = locations[1]
            pose.position.z_val = locations[2]
            pose.orientation.x_val = orientation[0]
            pose.orientation.y_val = orientation[1]
            pose.orientation.z_val = orientation[2]
            pose.orientation.w_val = orientation[3]
            return pose
            
    def __set_agent_location_and_target_point(self):
        pose = self.multirotor_client.simGetVehiclePose(vehicle_name = self.agent_name)           
        if self.agent_name == "SimpleFlight":
            locations = [0,-1,-5]
            pose = self.__set_pose(pose,locations,[0,0,0,1])
            self.multirotor_client.simSetVehiclePose(pose=pose, ignore_collision=True,vehicle_name = self.agent_name)
            self.__target_point = np.array([87.5,-1,-5]).astype(np.float32)
        elif self.agent_name == "Drone0":
            locations = [0,-1,-5]
            pose = self.__set_pose(pose,locations,[0,0,0,1])
            self.multirotor_client.simSetVehiclePose(pose=pose, ignore_collision=True,vehicle_name = self.agent_name)
            self.__target_point = np.array([87.5,-1,-5]).astype(np.float32)
        elif self.agent_name == "Drone1":
            locations = [0,-1,-5]
            pose = self.__set_pose(pose,locations,[0,0,0,1])
        
            self.multirotor_client.simSetVehiclePose(pose=pose, ignore_collision=True,vehicle_name = self.agent_name)
            self.__target_point = np.array([87.5,-1,-5]).astype(np.float32)
        elif self.agent_name == "Drone2":
            locations = [0,-1,-5]
            pose = self.__set_pose(pose,locations,[0,0,0,1])
        
            self.multirotor_client.simSetVehiclePose(pose=pose, ignore_collision=True,vehicle_name = self.agent_name)
            self.__target_point = np.array([87.5,-1,-5]).astype(np.float32)
        elif self.agent_name == "Drone3":
            locations = [0,-1,-5]
            pose = self.__set_pose(pose,locations,[0,0,0,1])
        
            self.multirotor_client.simSetVehiclePose(pose=pose, ignore_collision=True,vehicle_name = self.agent_name)
            self.__target_point = np.array([87.5,-1,-5]).astype(np.float32)
        else:
            locations = [0,-1,-5]
            pose = self.__set_pose(pose,locations,[0,0,0,1])
        
            self.multirotor_client.simSetVehiclePose(pose=pose, ignore_collision=True,vehicle_name = self.agent_name)
            self.__target_point = np.array([87.5,-1,-5]).astype(np.float32)

    def _calculate_distance_to_target(self):
        current_pos = self.multirotor_client.getMultirotorState(vehicle_name = self.agent_name).kinematics_estimated.position.to_numpy_array()
        return  np.abs(current_pos[0]-self.__target_point[0])
        
        
        
    def _do_action(self, action):
        """
        quad_offset = self.interpret_action(action)
        self.agent_vel.x_val = self.agent_vel.x_val  + quad_offset[0]
        self.agent_vel.y_val = self.agent_vel.y_val  + quad_offset[1]
        self.agent_vel.z_val = self.agent_vel.z_val  + quad_offset[2]
        
        self.drone_process = self.multirotor_client.moveByVelocityAsync(
                                self.agent_vel.x_val,
                                self.agent_vel.y_val,
                                self.agent_vel.z_val,
                                10,
                                drivetrain =airsim.DrivetrainType.ForwardOnly,
                                yaw_mode = airsim.YawMode(False,0),
                                vehicle_name = self.agent_name,
                            ).join()
        """
        #print(action)
        if action == 0:
            self.move_straight(0.2,20)
        elif action == 1:
            self.rotate_clockwise(0.2)
        elif action == 2:
            self.rotate_counterClockwise(0.2)
        else:
            print("No action")
        self.stop_drone()
            
    def move_straight(self,duration,speed):
        pitch, roll, yaw  = airsim.to_eularian_angles(self.multirotor_client.getMultirotorState(vehicle_name = self.agent_name).kinematics_estimated.orientation)
        vx = np.cos(yaw) * speed
        vy = np.sin(yaw) * speed
        self.multirotor_client.moveByVelocityZAsync(vx,vy,self.z,duration,
                                                    drivetrain = airsim.DrivetrainType.ForwardOnly,
                                                    vehicle_name = self.agent_name).join()

    def rotate_clockwise(self,duration):
        self.multirotor_client.rotateByYawRateAsync(150,duration,vehicle_name = self.agent_name).join()

    def rotate_counterClockwise(self,duration):
        self.multirotor_client.rotateByYawRateAsync(-150,duration,vehicle_name = self.agent_name).join()

    def stop_drone(self):
        self.multirotor_client.moveByVelocityZAsync(0,0,self.z,0.1,
                                                    drivetrain = airsim.DrivetrainType.ForwardOnly,
                                                    vehicle_name = self.agent_name).join()
        self.multirotor_client.rotateByYawRateAsync(0,0.1,vehicle_name = self.agent_name).join()


    def interpret_action(self, action):
        print(action, end='\r', flush=True)
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
        self.__target_point_list = [np.array([-25.00,28.00,-10]),
                                    np.array([-87.2,-47.9,-10]),
                                    np.array([70.6,0.3,-10]),
                                    np.array([193.5,-47.9,-10]),
                                    np.array([28.4,100,-10]),
                                    np.array([28.4,-139.4,-10]),
                                    np.array([-52.3,-113.1,-10]),
                                    np.array([193.5,107.9,-10]),
                                    np.array([125,-3,-10])]


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

    def quaternion_to_direction_vector(self, quat):
        # Extract the vector components (x, y, z)
        x, y, z = quat[1:]

        # Normalize the vector to get the direction
        direction_vector = np.array([x, y, z])
        direction_vector /= np.linalg.norm(direction_vector)

        return direction_vector

    def are_vectors_in_same_direction(self, vector1, vector2):
        dot_product = sum(x * y for x, y in zip(vector1, vector2))
        
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
        
        
        # Check for collision first
        if self.multirotor_client.simGetCollisionInfo(vehicle_name = self.agent_name).has_collided:
            # Set values for this agent and continue to the next agent
            #print("Collision detected")
            self.rewards = -10
            self.total_collision += 1
            print(self.total_collision)
            #self.terminations = True
            #return None

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
        print(dist)
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
        
        quad_vel_orgin = self.agent_vel
        quad_vel = [quad_vel_orgin.x_val, quad_vel_orgin.y_val, quad_vel_orgin.z_val]
        
        if np.absolute(quad_vel_orgin.x_val) >10 or np.absolute(quad_vel_orgin.y_val) >10 or np.absolute(quad_vel_orgin.z_val)>10:
            self.rewards += -((np.maximum(np.absolute(quad_vel_orgin.x_val),
                                          np.maximum(np.absolute(quad_vel_orgin.y_val), np.absolute(quad_vel_orgin.z_val)))
                               - SPEED_REDUCTION))/10
        else:
            self.rewards += ((np.maximum(np.absolute(quad_vel_orgin.x_val),
                                          np.maximum(np.absolute(quad_vel_orgin.y_val), np.absolute(quad_vel_orgin.z_val)))
                               - SPEED_REDUCTION))/10
        
        orientation = self.kinematics_estimated.orientation.to_numpy_array()
        vector_a= self.quaternion_to_direction_vector(orientation)
        vector_b = self.__target_point - self.agent_current_location
        result = self.are_vectors_in_same_direction(vector_a, vector_b)
        if result:
            self.rewards += 0.1
        '''
        #old_distance = np.linalg.norm(self.agent_old_locations - self.__target_point)
        #new_distance = np.linalg.norm(self.agent_current_location - self.__target_point)
        current_distance = self._calculate_distance_to_target()
        if current_distance < self.old_dist:
            self.rewards = 1 - current_distance/87.5
        elif  current_distance == self.old_dist:
            self.rewards = -0.1
        else:
            self.rewards = -1
    

        self.old_dist = current_distance

        #print(self.rewards, " + ", new_distance/87.5)
        
        #reward += 1 if orientation.x_val < 0.1 or orientation.y_val < 0.1 else 0
        if current_distance < 2:
            self.rewards += 100
            self.total_finish_eps +=1
            print("target aquire")
            print("total_finish_eps ", self.total_finish_eps)
        self.terminations = current_distance < 2
        #print(self.terminations)
        # Check truncation condition
        truncated = self.agent_current_step >= self.max_steps
        self.truncations = truncated
        self.accumulate_reward += self.rewards
        if self.accumulate_reward < -50:
            self.terminations = True
        
        return None
