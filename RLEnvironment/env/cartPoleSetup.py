import gymnasium as gym
import numpy as np
import torch
from collections import deque

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
        raw_rewards = {}
        
        # Track termination reason for better analysis
        if terminated:
            x, _, angle, _ = observation
            if abs(x) >= 2.4:
                info['termination_reason'] = 'position_limit'
            elif abs(angle) >= 0.209:
                info['termination_reason'] = 'angle_limit'
            else:
                info['termination_reason'] = 'other'
        
        # Initialize reward_scales if not exists
        if not hasattr(self, 'reward_scales'):
            self.reward_scales = {}
            
        # Initialize reward history for smoothing
        if not hasattr(self, 'reward_history'):
            self.reward_history = {name: [] for name in self.rewardComponents.keys()}
        
        for name, func in self.rewardComponents.items():
            if func and callable(func):
                try:
                    componentReward = func(observation, action)
                    raw_rewards[name] = componentReward
                    
                    # Add to history for smoothing
                    self.reward_history[name].append(componentReward)
                    if len(self.reward_history[name]) > 50:  # Keep last 50 values
                        self.reward_history[name].pop(0)
                    
                    # Update running scale estimate with more robust method
                    if name not in self.reward_scales:
                        self.reward_scales[name] = abs(componentReward)
                    else:
                        # More stable moving average using median filtering for outlier rejection
                        recent_rewards = self.reward_history[name][-10:] if len(self.reward_history[name]) >= 10 else self.reward_history[name]
                        median_reward = sorted(map(abs, recent_rewards))[len(recent_rewards)//2]
                        
                        # Slower adaptation rate for more stability (0.98/0.02 instead of 0.95/0.05)
                        self.reward_scales[name] = 0.98 * self.reward_scales[name] + 0.02 * median_reward
                    
                    # Normalize reward by its scale with improved numerical stability
                    scale = max(self.reward_scales[name], 1e-4)  # Ensure minimum scale to prevent large spikes
                    normalized_reward = componentReward / scale
                    
                    # Apply sigmoid normalization to contain extreme values
                    normalized_reward = 2.0 / (1.0 + np.exp(-normalized_reward)) - 1.0
                    
                    # Apply weight
                    weightedReward = normalized_reward * self.componentWeights[name]
                    
                    rewards.append(weightedReward)
                    info['componentRewards'][name] = componentReward
                    
                except Exception as e:
                    print(f"Error in reward component {name}: {e}")
                    rewards.append(0)
        
        reward = sum(rewards) if rewards else self.angleBasedReward(observation, action)
        
        # Add raw reward values to info for analysis
        info['raw_rewards'] = raw_rewards
        info['scales'] = {name: scale for name, scale in self.reward_scales.items()}
            
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

    def updateComponentWeight(self, componentNumber: int, weight: float, smooth_factor=0.05):
        """Update component weights with much smoother transitions and better bounds"""
        if componentNumber not in [1, 2]:
            raise ValueError("Only components 1 (stability) and 2 (efficiency) are supported")
            
        funcName = f'rewardFunction{componentNumber}'
        
        # Implement weight history tracking if not exists
        if not hasattr(self, 'weight_history'):
            self.weight_history = {
                'rewardFunction1': deque(maxlen=10),
                'rewardFunction2': deque(maxlen=10)
            }
            # Initialize with current weights
            self.weight_history['rewardFunction1'].append(self.componentWeights['rewardFunction1'])
            self.weight_history['rewardFunction2'].append(self.componentWeights['rewardFunction2'])
        
        # Check for too rapid changes (prevent oscillations)
        if len(self.weight_history[funcName]) >= 3:
            # Get last few weight changes
            recent_weights = list(self.weight_history[funcName])
            # If we're ping-ponging between values, slow down changes
            if (recent_weights[-1] > recent_weights[-2] and weight < recent_weights[-1]) or \
               (recent_weights[-1] < recent_weights[-2] and weight > recent_weights[-1]):
                # Reduce change magnitude
                smooth_factor *= 0.5
                print(f"Detected oscillation pattern. Reducing weight change rate.")
        
        # More restrictive bounds for stability vs efficiency
        if componentNumber == 1:  # Stability
            # Stability has higher minimum (0.3) to prevent instability
            weight = max(0.3, min(0.8, weight))
        else:  # Efficiency
            # Efficiency has lower maximum (0.7) to ensure enough stability 
            weight = max(0.2, min(0.7, weight))
        
        # Very smooth transition - use even smaller adjustment factor
        old_weight = self.componentWeights[funcName]
        new_weight = (1 - smooth_factor) * old_weight + smooth_factor * weight
        
        # Track weight history for oscillation detection
        self.weight_history[funcName].append(new_weight)
        
        # Apply new weight
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


