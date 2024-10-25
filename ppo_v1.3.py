import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision
import airsim
import time
import pickle

from typing import Callable
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space

from sb3_contrib import RecurrentPPO
import os

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.save_path_backup = os.path.join(log_dir, 'back_up')
        self.save_memory_path = 'tmp2/memory.pkl'
        self.load_memory_path = 'tmp2/memory.pkl'
        self.best_mean_reward = -np.inf
        self.rollout_buffer = None

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        
        '''
        time.sleep(0.5)
        #print(self.model.get_env().venv.venv.envs)
        #print(dir(self.model.get_env().venv.venv.envs))
        list_envs = self.model.get_env().venv.venv.envs
        i = 0
        for subenv in list_envs:
            current_env = subenv.env.env.env
            #start_time = time.time()
            current_env.drone_process.join()
            #print(i, " : ",time.time()-start_time)
            #i+=1
            
        current_env.multirotor_client.simPause(True)
        for subenv in list_envs:
            current_env = subenv.env.env.env
            current_env.kinematics_estimated = current_env.multirotor_client.getMultirotorState(vehicle_name = current_env.agent_name).kinematics_estimated
            current_env.agent_vel = current_env.kinematics_estimated.linear_velocity
            current_env.get_obs()
            current_env.compute_reward()
            current_env.agent_old_locations =  current_env.agent_current_location.copy()
        current_env.multirotor_client.simPause(False)
        '''   
        #print(dir(self.model.get_env().venv.venv.envs[0].env.env.env))
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
            
              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                    self.model.save("back_up")
          #self.model.save("current_learn")
          print(f"Finish saving")

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        #self.model.rollout_buffer.actions[-1,:] = 6
        #self.model.rollout_buffer.actions = np.roll(self.model.rollout_buffer.actions,shift=1,axis=0)      
        
    def custom_save_memory(self,rollout_buffer,observations, actions, rewards, episode_starts, values, log_probs):
        self.rollout_buffer = rollout_buffer
        #self.save_ppo_data(self.save_memory_path, observations, actions, rewards, episode_starts, values, log_probs)

    def custom_load_memory(self):
        return True
    '''
        is_empty,loaded_observations, loaded_actions, loaded_rewards, loaded_episode_starts, loaded_values, loaded_log_probs = self.load_ppo_data(self.load_memory_path)
        if not is_empty:
            self.rollout_buffer.add(
                                    loaded_observations,
                                    loaded_actions,
                                    loaded_rewards,
                                    loaded_episode_starts,
                                    loaded_values,
                                    loaded_log_probs,
                                )
        return is_empty
    '''
        
    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
            
              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                    self.model.save("back_up")
          #self.model.save("current_learn")
          print(f"Finish saving")
    
    def save_ppo_data(self,filename, observations, actions, rewards, episode_starts, values, log_probs):
        try:
            # Load existing data if the file already exists
            with open(filename, 'rb') as file:
                existing_data = pickle.load(file)
        except FileNotFoundError:
            existing_data = {}

        # Add new data
        data_to_save = {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'episode_starts': episode_starts,
            'values': values,
            'log_probs': log_probs,
        }

        # Update existing data with new data
        existing_data.update(data_to_save)

        # Save the combined data to the file
        with open(filename, 'wb') as file:
            pickle.dump(existing_data, file)

    def load_ppo_data(self,filename):
        try:
            with open(filename, 'rb') as file:
                # Check if the file is not empty
                if os.fstat(file.fileno()).st_size > 0:
                    loaded_data = pickle.load(file)
                    is_empty = False
                else:
                    loaded_data = {}
                    is_empty = True
        except (FileNotFoundError, EOFError):
            # Handle FileNotFoundError and EOFError together
            loaded_data = {}
            is_empty = True

        return (
            is_empty,
            loaded_data.get('observations', []),
            loaded_data.get('actions', []),
            loaded_data.get('rewards', []),
            loaded_data.get('episode_starts', []),
            loaded_data.get('values', []),
            loaded_data.get('log_probs', []),
        )

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return 0.01 * initial_value

    return func

# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

if __name__ == '__main__':
    num_envs = 0 # Set the number of environments
    env_id = "Step1-v0"
    #"AutoDrone-v0"
    #"MultiDrone-v0"
    #"Step1-v0"
    
    # Connect to airsim
    __multirotor_client = airsim.MultirotorClient()
    __multirotor_client.confirmConnection()
    __multirotor_client.enableApiControl(True)
    __multirotor_client.armDisarm(True)
    __multirotor_client.simPause(False)
    exist_drone_name = __multirotor_client.listVehicles()
    __multirotor_client.takeoffAsync(timeout_sec = 6)
    print(exist_drone_name)
    
    agents = ["SimpleFlight"]
    for i in range(num_envs):
        agent_name = "Drone" + str(i)
        agents.append(agent_name)
        if agent_name not in exist_drone_name:
            pose = airsim.Pose(airsim.Vector3r(i-10, i-10, -2), airsim.to_quaternion(0, 0, 0))
            __multirotor_client.simAddVehicle(agent_name, "simpleflight", pose) 
        __multirotor_client.enableApiControl(True,vehicle_name = agent_name)
        __multirotor_client.armDisarm(True,vehicle_name = agent_name)
        #__multirotor_client.takeoffAsync(timeout_sec = 2, vehicle_name = agent_name)

    
    env = SubprocVecEnv([lambda agent_name=agent_name: gym.make(env_id) for agent_name in agents])
    #env = DummyVecEnv([lambda agent_name=agent_name: gym.make(env_id) for agent_name in agents])

    #env = VecFrameStack(env,4)
    """
    env_vec = VecMonitor(env, log_dir)
    
    if False:
        model_new = PPO('MultiInputPolicy',
                        env=env_vec, verbose=1,
                        tensorboard_log="./board/",
                        learning_rate=linear_schedule(0.0005),
                        ent_coef=0.001,
                        n_steps=256,
                        batch_size=128,
                        n_epochs=10,
                        policy_kwargs = policy_kwargs,)
    else:
        print("Load model")
        custom_objects = {'learning_rate':linear_schedule(0.001),
                      'ent_coef':0.0001,
                      'n_steps':1024,
                      'batch_size':64,
                      'n_epochs':10,
                        }
        
        model_old = PPO.load('current_learn.zip',custom_objects=custom_objects)
        
        #print(model_old_parameter['policy'].keys())
        #del model_old_parameter['policy.optimizer']
        #del model_old_parameter['policy']['action_net.weight']
        #del model_old_parameter['policy']['action_net.bias']
        model_old_parameter = model_old.get_parameters()
        
        del model_old_parameter['policy']['features_extractor.extractors.image.cnn.0.weight']
        del model_old_parameter['policy']['pi_features_extractor.extractors.image.cnn.0.weight']
        del model_old_parameter['policy']['vf_features_extractor.extractors.image.cnn.0.weight']
        
        #print(model_old_parameter)
        #del model_old_parameter['policy']['mlp_extractor.value_net.4.bias']
        #del model_old_parameter['policy.optimizer']
        model_new = PPO('MultiInputPolicy',
                        env=env_vec, verbose=1,
                        tensorboard_log="./board/",
                        learning_rate=linear_schedule(0.001),
                        ent_coef=0.1,
                        n_steps=512,
                        batch_size=64,
                        n_epochs=10,
                        )
        model_new.set_parameters(model_old_parameter, exact_match= False)
        
        #ppo_drone.zip
        #tmp/back_up.zip
        #model_new = PPO.load('ppo_drone.zip',env=env_vec,custom_objects=custom_objects)

    callback = SaveOnBestTrainingRewardCallback(check_freq=512, log_dir=log_dir,)
    print("------------- Start Learning -------------")
    model_new.learn(total_timesteps=500000,callback=callback, tb_log_name="PPO-00003")
    model_new.save('current_learn')
        
    #ppo_buffer_memory = model_new.rollout_buffer
    #print(model_new.policy)
    """
    x = 0.0003
    y = 0.1
    
    while True:
        env1 = VecMonitor(env, log_dir)
        
        print("Load model")
        custom_objects = {'learning_rate':linear_schedule(0.01),
                      'ent_coef':0.1,
                      'n_steps':512,
                      'batch_size':128,
                      'n_epochs':20,
                          'gamma':0.1,
                          'gae_lambda':0.65,
                          'clip_range':linear_schedule(0.2)
                        }
        #ppo_drone.zip
        #tmp/back_up.zip
        '''
        model_new = PPO('MlpPolicy',
                        env=env1, verbose=1,
                        tensorboard_log="./board/",
                        learning_rate=linear_schedule(0.01),
                        ent_coef=0.01,
                        n_steps=512,
                        batch_size=128,
                        n_epochs=20,
                        )
        '''
        model_new = PPO.load('ppo_drone_landing.zip',env=env1,custom_objects=custom_objects)
        
        callback = SaveOnBestTrainingRewardCallback(check_freq=512  , log_dir=log_dir,)
        print("------------- Start Learning -------------")
        model_new.learn(total_timesteps=100000 ,callback=callback, tb_log_name="PPO-00003")
        model_new.save('ppo_drone_landing')
        x*=0.9
        y*=0.9 
    print("------------- Done Learning -------------")
    '''
    obs, infos = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, terminations, truncations, infos = env.step(action)
    '''
