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


import re

def extractFunctionCode(responseString):
    # Updated pattern to match from the first 'def' to the end of the string
    function_pattern = r"(def\s+dynamicRewardFunction\(.*\):[\s\S]*)"
    match = re.search(function_pattern, responseString)

    if not match:
        raise ValueError("No valid function definition found in the response.")

    functionString = match.group(1)
    return functionString.strip()


def setRewardFunction(functionString):
    localNamespace = {}
    try:
        # Extract function code from response string
        functionCode = extractFunctionCode(functionString)  # Extract only the function
        exec(functionCode, globals(), localNamespace)
    except Exception as e:
        raise ValueError(f"Failed to execute function string: {e}")

    newFunction = None
    for item in localNamespace.values():
        if callable(item):
            newFunction = item
            break

    if newFunction is None:
        raise ValueError("No valid function was extracted from the response.")
    
    return newFunction


# -- Adaptive Reward Function
def updateRewardFunction(env, updateSystem, episode):
    print(f"\nAttempting reward function update at episode {episode}")
    funcName = f'rewardFunction_{updateSystem.targetComponent}'
    currentFunc = env.rewardComponents[funcName]
    newFunction, updated = updateSystem.validateAndUpdate(currentFunc)

    if updated:
        print(f"\nUpdating reward component {updateSystem.targetComponent}...")
        env.setComponentReward(updateSystem.targetComponent, newFunction)
        updateSystem.lastUpdateEpisode = episode
    else:
        print("\nKeeping current reward function.")



from AdaptiveRewardFunctionLearning.RewardGeneration.rewardCritic import RewardUpdateSystem
from AdaptiveRewardFunctionLearning.RewardGeneration.rewardCodeGeneration import analyseFailure, updateComponentWeight

def updateCompositeRewardFunction(env, updateSystem, metrics, dynamicRewardFunction):
    """
    Updates composite reward function components based on waiting time and performance
    
    Args:
        env: The environment instance
        updateSystem: System handling LLM updates
        metrics: Dictionary containing current performance metrics
        dynamicRewardFunction: The dynamic reward function being used
    """
    print(f"\nChecking composite components for updates at episode {metrics['currentEpisode']}")
    
    # Check each composite component
    for componentName in ['stability', 'efficiency', 'time']:
        # Skip if component doesn't exist in weights
        if not hasattr(dynamicRewardFunction, 'weights') or componentName not in dynamicRewardFunction.weights:
            continue
            
        # Check if this component should be updated
        if updateSystem.waitingTime(componentName, metrics, 
                      dynamicRewardFunction.weights[componentName]['lastUpdate'],
                      threshold=0.5):
            
            print(f"\nAttempting to update composite component: {componentName}")
            
            # Analyze failure type for weight adjustment
            failureType = analyseFailure(dynamicRewardFunction.lastObservation)
            
            # Update component weight
            oldWeight = dynamicRewardFunction.weights[componentName]['value']
            updateComponentWeight(componentName, failureType)
            newWeight = dynamicRewardFunction.weights[componentName]['value']
            
            # Log the update
            if not hasattr(dynamicRewardFunction, 'compositeHistory'):
                dynamicRewardFunction.compositeHistory = []
            
            dynamicRewardFunction.compositeHistory.append({
                'component': componentName,
                'episode': metrics['currentEpisode'],
                'oldWeight': oldWeight,
                'newWeight': newWeight,
                'failureType': failureType,
                'metrics': metrics.copy()
            })
            
            # Update last update time
            dynamicRewardFunction.weights[componentName]['lastUpdate'] = metrics['currentEpisode']
            
            print(f"Updated {componentName} weight: {oldWeight:.3f} -> {newWeight:.3f}")