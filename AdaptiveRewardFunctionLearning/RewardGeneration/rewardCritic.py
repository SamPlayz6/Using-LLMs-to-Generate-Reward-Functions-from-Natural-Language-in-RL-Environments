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
        
    def recordEpisode(self, info: dict, steps: int, totalReward: float):
        """Record both reward and episode information"""
        if 'componentRewards' in info:
            componentReward = info['componentRewards'].get(f'rewardFunction{self.targetComponent}')
            if componentReward is not None:
                self.rewardHistory.append(componentReward)
                self.episodeLengths.append(steps)
        
    def validateAndUpdate(self, currentFunction: str):
        try:
            print(f"\nGenerating new reward function for component {self.targetComponent}...")
            graph = self.generatePerformanceGraphs()
            updatePrompt = self.createUpdatePrompt(currentFunction, graph)

            #queryAnthropicApi
            from AdaptiveRewardFunctionLearning.Prompts.APIQuery import queryAnthropicApi
            proposedFunction = queryAnthropicApi(self.apiKey, self.modelName, updatePrompt)
            
            print("\nProposed Function:")
            print(proposedFunction)
            
            print("\nWaiting 10 seconds before critic evaluation...")
            time.sleep(10)
            
            print("\nGetting critic's evaluation...")
            criticPrompt = self.createCriticPrompt(proposedFunction)
            criticResponse = queryAnthropicApi(self.apiKey, self.modelName, criticPrompt)
            
            print("\nCritic Response:")
            print(criticResponse)
            
            approved = criticResponse.strip().endswith("Decision: Yes")
            print(f"\nCritic Decision: {'Approved' if approved else 'Rejected'}")
            
            if approved:
                self.lastRewardFunction = currentFunction
                return proposedFunction, True
            return currentFunction, False
            
        except Exception as e:
            print(f"\nError during update: {e}")
            return currentFunction, False

    def waitingTime(self, episode: int):
        UPDATE_INTERVAL = 100
        return episode - self.lastUpdateEpisode >= UPDATE_INTERVAL