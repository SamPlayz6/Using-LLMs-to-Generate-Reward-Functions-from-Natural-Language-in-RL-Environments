class RewardFunctionWrapper:
    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        
        if self.using_components and any(self.reward_components.values()):
            info['component_rewards'] = {}
            rewards = []
            for name, func in self.reward_components.items():
                if func and callable(func):
                    component_reward = func(observation, action)
                    rewards.append(component_reward)
                    info['component_rewards'][name] = component_reward
            reward = sum(rewards) / len(rewards) if rewards else self.rewardFunction(observation, action)
        else:
            if self.rewardFunction is None or not callable(self.rewardFunction):
                print("Warning: rewardFunction is None or not callable.")
                self.rewardFunction = self.angleBasedReward
            reward = self.rewardFunction(observation, action)
            
        return observation, reward, terminated, truncated, info

    def LLMRewardFunction(self, functionString):
        localNamespace = {}
        try:
            exec(functionString, globals(), localNamespace)
            new_function = None
            for item in localNamespace.values():
                if callable(item):
                    new_function = item
                    break
            if new_function is None:
                raise ValueError("Extracted function is not callable.")
            self.setRewardFunction(new_function)
            print("Reward function successfully updated.")
        except Exception as e:
            print(f"Failed to execute function string: {e}")
            self.setRewardFunction(self.angleBasedReward)

    def setRewardFunction(self, rewardFunction):
        self.rewardFunction = rewardFunction

    def updateRewardFunction(self, functionString):
        print("updateReward Function: " + functionString)
        try:
            newFunction = setRewardFunction(functionString)
            if newFunction and callable(newFunction):
                self.setRewardFunction(newFunction)
                print("Reward function updated dynamically from LLM.")
            else:
                raise ValueError("Extracted function is not callable.")
        except Exception as e:
            print(f"Failed to update reward function: {e}")
            self.setRewardFunction(self.angleBasedReward)