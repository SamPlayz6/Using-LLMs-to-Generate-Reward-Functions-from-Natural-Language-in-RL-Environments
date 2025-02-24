import gymnasium as gym
import numpy as np
import torch

class CustomCartPoleEnv(gym.Wrapper):
    def __init__(self, env, numComponents=2):  # Default to 2 components now
        super().__init__(env)
        self.env = env
        self.rewardFunction = self.angleBasedReward
        
        # Component handling
        self.usingComponents = True  # Always use components now
        self.rewardComponents = {}
        self.componentWeights = {}
        
        # Initialize two components with default weights
        self.rewardComponents['rewardFunction1'] = None  # Stability
        self.rewardComponents['rewardFunction2'] = None  # Efficiency
        self.componentWeights['rewardFunction1'] = 0.6   # Higher weight for stability
        self.componentWeights['rewardFunction2'] = 0.4   # Lower weight for efficiency

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        
        info['componentRewards'] = {}
        rewards = []
        
        # Initialize reward_scales if not exists
        if not hasattr(self, 'reward_scales'):
            self.reward_scales = {}
        
        for name, func in self.rewardComponents.items():
            if func and callable(func):
                try:
                    componentReward = func(observation, action)
                    
                    # Update running scale estimate
                    if name not in self.reward_scales:
                        self.reward_scales[name] = abs(componentReward)
                    else:
                        # Exponential moving average for scale
                        self.reward_scales[name] = 0.95 * self.reward_scales[name] + 0.05 * abs(componentReward)
                    
                    # Normalize reward by its scale
                    normalized_reward = componentReward / (self.reward_scales[name] + 1e-8)
                    weightedReward = normalized_reward * self.componentWeights[name]
                    
                    rewards.append(weightedReward)
                    info['componentRewards'][name] = componentReward
                    
                except Exception as e:
                    print(f"Error in reward component {name}: {e}")
                    rewards.append(0)
        
        reward = sum(rewards) if rewards else self.angleBasedReward(observation, action)
            
        return observation, reward, terminated, truncated, info

    def setComponentReward(self, componentNumber: int, rewardFunction):
        if componentNumber not in [1, 2]:  # Only allow components 1 or 2
            raise ValueError("Only components 1 (stability) and 2 (efficiency) are supported")
            
        funcName = f'rewardFunction{componentNumber}'
        if funcName in self.rewardComponents:
            self.rewardComponents[funcName] = rewardFunction
            if componentNumber == 1:  # Stability component is primary
                self.rewardFunction = rewardFunction
            return True
        return False

    def updateComponentWeight(self, componentNumber: int, weight: float, smooth_factor=0.1):
        """Update component weights with smoother transitions"""
        if componentNumber not in [1, 2]:
            raise ValueError("Only components 1 (stability) and 2 (efficiency) are supported")
            
        funcName = f'rewardFunction{componentNumber}'
        
        # Ensure weight is within bounds
        weight = max(0.2, min(0.8, weight))  # Bound weights between 0.2 and 0.8
        
        # Smooth transition
        old_weight = self.componentWeights[funcName]
        new_weight = (1 - smooth_factor) * old_weight + smooth_factor * weight
        self.componentWeights[funcName] = new_weight
        
        # Update other component to maintain sum of 1.0
        other_name = 'rewardFunction2' if componentNumber == 1 else 'rewardFunction1'
        self.componentWeights[other_name] = 1.0 - new_weight
        
        print(f"Updated weights - Stability: {self.componentWeights['rewardFunction1']:.3f}, "
              f"Efficiency: {self.componentWeights['rewardFunction2']:.3f}")
        return True



#---- ??


    def angleBasedReward(self, observation, action):
        """Default reward function if no components are set"""
        _, _, angle, _ = observation
        return np.cos(angle)

    def setEnvironmentParameters(self, masscart=1.0, length=1.0, gravity=9.8):
        """Update environment physical parameters"""
        self.env.masscart = masscart
        self.env.length = length
        self.env.gravity = gravity
        print(f"Environment parameters updated: masscart={masscart}, length={length}, gravity={gravity}")

    def setRewardFunction(self, rewardFunction):
        """Set a single reward function (used mainly for baseline comparison)"""
        self.rewardFunction = rewardFunction
        
    def getCurrentWeights(self):
        """Get current component weights for monitoring"""
        return {
            'stability': self.componentWeights['rewardFunction1'],
            'efficiency': self.componentWeights['rewardFunction2']
        }


