import gymnasium as gym
import numpy as np
import torch

class CustomCartPoleEnv(gym.Wrapper):
    def __init__(self, env, numComponents=None):
        super().__init__(env)
        self.env = env
        self.rewardFunction = self.angleBasedReward
        
        # Component handling
        self.usingComponents = bool(numComponents)
        self.rewardComponents = {}
        self.componentWeights = {}
        
        if self.usingComponents:
            for i in range(1, numComponents + 1):
                self.rewardComponents[f'rewardFunction{i}'] = None
                self.componentWeights[f'rewardFunction{i}'] = 1.0/numComponents

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        
        if self.usingComponents and any(self.rewardComponents.values()):
            info['componentRewards'] = {}
            rewards = []
            
            for name, func in self.rewardComponents.items():
                if func and callable(func):
                    try:
                        componentReward = func(observation, action)
                        weightedReward = componentReward * self.componentWeights[name]
                        rewards.append(weightedReward)
                        info['componentRewards'][name] = componentReward
                    except Exception as e:
                        print(f"Error in reward component {name}: {e}")
                        rewards.append(0)
            
            reward = sum(rewards) if rewards else self.angleBasedReward(observation, action)
        else:
            if self.rewardFunction is None or not callable(self.rewardFunction):
                print("Warning: rewardFunction is None or not callable.")
                self.rewardFunction = self.angleBasedReward
            reward = self.rewardFunction(observation, action)
            
        return observation, reward, terminated, truncated, info

    def setComponentReward(self, componentNumber: int, rewardFunction):
        if not self.usingComponents:
            raise ValueError("Environment not initialized for components")
            
        funcName = f'rewardFunction{componentNumber}'
        if funcName in self.rewardComponents:
            self.rewardComponents[funcName] = rewardFunction
            if componentNumber == 1:
                self.rewardFunction = rewardFunction
            return True
        return False

    def updateComponentWeight(self, componentNumber: int, weight: float):
        funcName = f'rewardFunction{componentNumber}'
        if funcName in self.componentWeights:
            self.componentWeights[funcName] = weight
            total = sum(self.componentWeights.values())
            for name in self.componentWeights:
                self.componentWeights[name] /= total
            return True
        return False


#---- ??


    def angleBasedReward(self, observation, action):
        _, _, angle, _ = observation
        return np.cos(angle)

    def setEnvironmentParameters(self, masscart=1.0, length=1.0, gravity=9.8):
        self.env.masscart = masscart
        self.env.length = length
        self.env.gravity = gravity
        print(f"Environment parameters updated: masscart={masscart}, length={length}, gravity={gravity}")


    def setRewardFunction(self, rewardFunction):
        self.rewardFunction = rewardFunction


