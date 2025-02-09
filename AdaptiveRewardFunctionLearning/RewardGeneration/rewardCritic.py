import re
import matplotlib.pyplot as plt
import time
import base64
from collections import deque
import io
import numpy as np

def extractFunctionCode(responseString):
    # First try to find code between triple backticks
    code_pattern = r"```python\n(def\s+\w+\(.*?\)[\s\S]*?)\n```"
    match = re.search(code_pattern, responseString)
    
    if match:
        code = match.group(1)
        # Ensure function name matches target
        code = re.sub(r"def\s+\w+\(", "def rewardFunction", code)
        return code.strip()
        
    # Fallback to finding raw function definition
    function_pattern = r"(def\s+\w+\(.*\):[\s\S]*)"
    match = re.search(function_pattern, responseString)
    
    if not match:
        raise ValueError("No valid function definition found in the response.")

    functionString = match.group(1)
    return functionString.strip()

class RewardUpdateSystem:
    def __init__(self, apiKey: str, modelName: str, maxHistoryLength: int = 100, targetComponent: int = 1):
        self.apiKey = apiKey
        self.modelName = modelName
        self.targetComponent = targetComponent
        
        # Performance tracking
        self.rewardHistory = deque(maxlen=maxHistoryLength)
        self.episodeLengths = deque(maxlen=maxHistoryLength)
        self.episodeCount = 0
        self.lastUpdateEpisode = 0
        self.lastRewardFunction = None
        
    def generatePerformanceGraphs(self):
        """Generate graphs for both rewards and episode lengths"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot rewards
        ax1.plot(list(self.rewardHistory), label=f'Reward Component {self.targetComponent}')
        ax1.set_title('Reward History')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Reward Value')
        ax1.grid(True)
        
        # Plot episode lengths
        ax2.plot(list(self.episodeLengths), label='Episode Length', color='green')
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps until fall')
        ax2.grid(True)
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode()

    def createUpdatePrompt(self, currentFunction: str, graph: str):
        recentRewards = list(self.rewardHistory)[-50:] if self.rewardHistory else []
        recentLengths = list(self.episodeLengths)[-50:] if self.episodeLengths else []
        
        return [{
            "role": "user",
            "content": f"""Analyze this reward component function and its performance metrics.
            
            Current Function:
            {currentFunction}
            
            Performance Metrics:
            - Average Reward: {np.mean(recentRewards) if recentRewards else 'No data'}
            - Average Episode Length: {np.mean(recentLengths) if recentLengths else 'No data'} steps
            - Max Episode Length: {max(recentLengths) if recentLengths else 'No data'} steps
            - Min Episode Length: {min(recentLengths) if recentLengths else 'No data'} steps
            
            Performance Graphs:
            {graph}
            
            Previous Function:
            {self.lastRewardFunction if self.lastRewardFunction else 'No previous function'}
            
            Requirements:
            1. The function MUST be named 'rewardFunction{self.targetComponent}'
            2. Focus on both reward optimization and episode duration
            3. Include detailed inline comments explaining changes
            4. Compare old vs new values in comments
            
            Output only the modified function with detailed inline comments."""
        }]

    def createCriticPrompt(self, proposedFunction: str):
        return [{
            "role": "user",
            "content": f"""Act as a reward function critic. Analyze this proposed reward component:

            {proposedFunction}

            Evaluate:
            1. Is the function properly named 'rewardFunction{self.targetComponent}'?
            2. Are reward calculations mathematically sound?
            3. Does it focus on both reward quality and episode duration?
            4. Are there any potential issues?
            
            End your response with EXACTLY one line containing only "Decision: Yes" or "Decision: No"."""
        }]
        
    def recordEpisode(self, info, steps, totalReward):
        """Record episode information and maintain debug metrics"""
        if not hasattr(self, 'episode_history'):
            self.episode_history = []
        
        # Store episode information
        self.episode_history.append({
            'info': info,
            'steps': steps,
            'reward': totalReward,
            'episode': len(self.episode_history)  # Add episode counter
        })
        
        # Debug print every 1000 episodes
        if len(self.episode_history) % 1000 == 0:
            print(f"\nDebug Metrics at episode {len(self.episode_history)}:")
            print(f"Recent average reward: {np.mean([e['reward'] for e in self.episode_history[-100:]]):.2f}")
            print(f"Recent average steps: {np.mean([e['steps'] for e in self.episode_history[-100:]]):.2f}")
            print(f"Last update episode: {self.lastUpdateEpisode}")
        
    def validateAndUpdate(self, currentFunction: str):
        try:
            print(f"\nGenerating new reward function for component {self.targetComponent}...")
            graph = self.generatePerformanceGraphs()
            updatePrompt = self.createUpdatePrompt(currentFunction, graph)
    
            proposedFunction = queryAnthropicApi(self.apiKey, self.modelName, updatePrompt)
            
            print("\nProposed Function:")
            print(proposedFunction)
            
            # Extract the actual function code
            newFunctionCode = extractFunctionCode(proposedFunction)
            
            criticResponse = queryAnthropicApi(self.apiKey, self.modelName, criticPrompt)
            approved = criticResponse.strip().endswith("Decision: Yes")
            
            if approved:
                self.lastRewardFunction = currentFunction
                self.logFunctionUpdate(f'Component {self.targetComponent}', currentFunction, newFunctionCode)
                # Store the new function code
                if not hasattr(dynamicRewardFunction, 'function_updates'):
                    dynamicRewardFunction.function_updates = {}
                dynamicRewardFunction.function_updates[f'rewardFunction{self.targetComponent}'] = newFunctionCode
                
                # Make sure to record when the change happened
                if not hasattr(dynamicRewardFunction, 'rewardChanges'):
                    dynamicRewardFunction.rewardChanges = []
                dynamicRewardFunction.rewardChanges.append(self.episodeCount)
                
                return newFunctionCode, True
                
            return currentFunction, False
                
        except Exception as e:
            print(f"\nError during update: {e}")
            return currentFunction, False


    def logFunctionUpdate(self, component, old_func, new_func):
        """Log when a reward function is updated"""
        print(f"\nUpdating {component} reward function:")
        print("Old function:")
        print(old_func)
        print("\nNew function:")
        print(new_func)
        print("-" * 50)


# ----- Watining Time Function


    def waitingTime(self, componentName, metrics, lastUpdateEpisode, threshold=100):
    currentEpisode = metrics['currentEpisode']
    timeSinceUpdate = currentEpisode - lastUpdateEpisode
    
    # Get recent performance trends
    recentRewards = metrics['recentRewards']
    
    # Calculate stability metrics
    avgStandTime = metrics['averageBalanceTime']
    standTimeVariance = metrics['balanceTimeVariance']
    
    # Component-specific thresholds with increased sensitivity
    if componentName == 'stability':
        variance_threshold = threshold * 0.6  # More sensitive to variance
    elif componentName == 'efficiency':
        variance_threshold = threshold * 1.4  # Less sensitive to variance
    else:
        variance_threshold = threshold
    
    # Update conditions
    if timeSinceUpdate < 500:  # Increased minimum time between updates
        return False
        
    if len(recentRewards) > 75:
        rewardDerivative = np.gradient(recentRewards)

        # Calculate different trend windows
        short_trend = np.mean(rewardDerivative[-20:])
        medium_trend = np.mean(rewardDerivative[-40:])
        
        # Calculate performance stability
        current_performance = np.mean(recentRewards[-20:])
        performance_variance = np.var(recentRewards[-20:])
        long_term_performance = np.mean(recentRewards[-100:])
        
        # Different conditions for updates:
        should_update = False
        
        # 1. Major performance drop with high variance
        if timeSinceUpdate >= 1000 and \
        performance_variance > current_performance * 0.5 and \
        short_trend > current_performance * 1.2:
            print(f"\nHigh variance with performance drop detected for {componentName}")
            print(f"Variance: {performance_variance:.2f}, Current Performance: {current_performance:.2f}")
            should_update = True
        
        # 2. Sustained poor performance
        elif timeSinceUpdate >= 2000 and \
            short_trend > current_performance * 1.15 and \
            medium_trend > current_performance * 1.1:
            print(f"\nSustained performance decline detected for {componentName}")
            print(f"Short trend: {short_trend:.2f}, Medium trend: {medium_trend:.2f}")
            should_update = True
            
        # 3. Significant performance degradation
        elif current_performance < 0.5 * long_term_performance:
            print(f"\nSignificant performance degradation detected for {componentName}")
            print(f"Current: {current_performance:.2f} vs Long-term: {long_term_performance:.2f}")
            should_update = True
        
        return should_update
        
    return False


    # def waitingTime(self, componentName, metrics, lastUpdateEpisode, threshold=5000):
    #     """
    #     Determine if it's time to update the reward function
    #     """
    #     currentEpisode = metrics['currentEpisode']
    #     timeSinceUpdate = currentEpisode - lastUpdateEpisode
        
    #     # Only return True at exact intervals
    #     should_update = (currentEpisode % threshold == 0 and currentEpisode != 0)
        
    #     # Only print debug info when we're actually updating or at major intervals
    #     if should_update or currentEpisode % 2000 == 0:
    #         print(f"\nChecking for update at episode {currentEpisode}, time since last update: {timeSinceUpdate}")
        
    #     if should_update:
    #         print(f"✓ Update triggered at episode {currentEpisode}")
        
    #     return should_update