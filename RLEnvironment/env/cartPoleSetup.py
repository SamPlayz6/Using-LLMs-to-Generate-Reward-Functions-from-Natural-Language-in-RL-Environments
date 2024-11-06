import gymnasium as gym
import numpy as np
import torch

class CustomCartPoleEnv(gym.Wrapper):
    def __init__(self, env, num_components=None):
        super().__init__(env)
        self.env = env
        self.rewardFunction = self.angleBasedReward
        
        self.using_components = bool(num_components)
        self.reward_components = {}
        if self.using_components:
            for i in range(1, num_components + 1):
                self.reward_components[f'reward_function_{i}'] = None

    def angleBasedReward(self, observation, action):
        _, _, angle, _ = observation
        return np.cos(angle)

    def setEnvironmentParameters(self, masscart=1.0, length=1.0, gravity=9.8):
        self.env.masscart = masscart
        self.env.length = length
        self.env.gravity = gravity
        print(f"Environment parameters updated: masscart={masscart}, length={length}, gravity={gravity}")