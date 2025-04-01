import re
import matplotlib.pyplot as plt
import time
import base64
from collections import deque
import io
import numpy as np
import anthropic

from .rewardCodeGeneration import dynamicRewardFunction


def queryAnthropicApi(api_key, model_name, messages, max_tokens=1024):
    client = anthropic.Anthropic(api_key=api_key)
    generatedRewardFunction = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=messages
    )
    return generatedRewardFunction.content[0].text

def queryAnthropicExplanation(api_key, model_name, explanation_message, max_tokens=1024):
    client = anthropic.Anthropic(api_key=api_key)
    explanationResponse = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=explanation_message
    )
    return explanationResponse.content[0].text



    # ---------------------

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
    def __init__(self, apiKey: str, modelName: str, maxHistoryLength: int = 100, targetComponent: int = 1, max_updates_per_run: int = 3):
        self.apiKey = apiKey
        self.modelName = modelName
        self.targetComponent = targetComponent
        
        # Performance tracking
        self.rewardHistory = deque(maxlen=maxHistoryLength)
        self.episodeLengths = deque(maxlen=maxHistoryLength)
        self.episodeCount = 0
        self.lastUpdateEpisode = 0
        
        # Update limits to control API usage
        self.max_updates_per_run = max_updates_per_run
        self.update_counts = {1: 0, 2: 0}  # Track updates per component
        
        # Memory of reward functions
        self.bestRewardFunction = None
        self.bestPerformance = float('-inf')
        self.rewardFunctionHistory = []
        self.performanceHistory = []
        self.cooldownPeriod = 1500  # Episodes to wait after update
        self.evaluationWindow = 50  # Episodes to evaluate performance
        
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
        componentType = "stability" if self.targetComponent == 1 else "efficiency"
        recentRewards = list(self.rewardHistory)[-50:] if self.rewardHistory else []
        
        return [{
            "role": "user",
            "content": f"""Analyze this {componentType} reward component function and its performance metrics.
                
            CRITICAL PRIORITY STRUCTURE:
            1. Pole angle stability MUST be the primary reward component with the HIGHEST POSITIVE WEIGHT
            2. Cart position and velocity should only be secondary penalty terms with smaller weights
            3. The sum of all penalty weights must never exceed 50% of the stability reward weight
                
            IMPORTANT CONSTRAINTS:
            1. The function MUST have signature: def reward_function(observation, action)
            2. The state variables available in 'observation' are:
               - observation[0]: Cart Position
               - observation[1]: Cart Velocity
               - observation[2]: Pole Angle
               - observation[3]: Pole Angular Velocity
            3. The 'action' is a SCALAR (not a list) with value 0 or 1
            4. DO NOT reference 'next_state' or any variables not in these lists
            5. Keep the function SIMPLE with MAX 3 components
            6. Avoid excessive scaling factors
            
            WEIGHT GUIDELINES:
            - Angle stability: Use positive weight between 1.5-3.0
            - Position penalty: Use negative weight between -0.1 to -0.5
            - Velocity penalty: Use negative weight between -0.1 to -0.3
            
            Current Function:
            {currentFunction}
            
            Performance Metrics:
            - Average Reward: {np.mean(recentRewards) if recentRewards else 'No data'}
            - Historical Best: {self.bestPerformance if self.bestPerformance != float('-inf') else 'No data'}
            
            Requirements:
            1. Make only SMALL, incremental improvements
            2. Focus on {componentType} aspects
            3. Include detailed inline comments
            4. VERIFY the function only uses observation and action as per constraints
            5. Ensure the reward is POSITIVE when the pole is upright, even if the cart is off-center
            
            Output only the modified function with detailed inline comments."""
        }]
    
    def _format_performance_history(self):
        if not self.performanceHistory:
            return "No previous updates"
        
        history = "\n".join([
            f"Episode {p['episode']}: Performance {p['performance']:.2f} (Best: {p['best']:.2f})"
            for p in self.performanceHistory[-3:]  # Show last 3 updates
        ])
        return history
    
    def createCriticPrompt(self, proposedFunction: str):
        componentType = "stability" if self.targetComponent == 1 else "efficiency"
        
        return [{
            "role": "user",
            "content": f"""Act as a reward function critic for a cart-pole environment. Analyze this proposed {componentType} reward component:
    
            {proposedFunction}
    
            BASIC REQUIREMENTS:
            1. Function name must be 'rewardFunction{self.targetComponent}'
            2. Must accept observation and action parameters
            3. Must return a numerical reward
    
            YOUR EVALUATION GUIDELINES:
            - Be lenient and approve functions that satisfy the basic requirements
            - Focus on whether the function will work at all, not if it's optimal
            - Approve functions even if they have minor issues
            - Reject only if there are critical errors that would prevent operation
            
            Evaluate:
            1. Does it have the correct function name?
            2. Does it accept the right parameters?
            3. Does it calculate some kind of sensible reward?
            
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
            'episode': len(self.episode_history)
        })
        
        # Debug print every 1000 episodes
        if len(self.episode_history) % 1000 == 0:
            print(f"\nDebug Metrics at episode {len(self.episode_history)}:")
            print(f"Recent average reward: {np.mean([e['reward'] for e in self.episode_history[-100:]]):.2f}")
            print(f"Recent average steps: {np.mean([e['steps'] for e in self.episode_history[-100:]]):.2f}")
            print(f"Last update episode: {self.lastUpdateEpisode}")
        
    def validateAndUpdate(self, currentFunction: str):
        try:
            # Check if we've reached the update limit for this component
            if self.update_counts.get(self.targetComponent, 0) >= self.max_updates_per_run:
                print(f"\nReached maximum updates ({self.max_updates_per_run}) for component {self.targetComponent}. Skipping update.")
                return currentFunction, False
            
            componentType = "stability" if self.targetComponent == 1 else "efficiency"
            print(f"\nGenerating new {componentType} reward function...")
            print(f"Update count: {self.update_counts.get(self.targetComponent, 0)}/{self.max_updates_per_run}")
            
            graph = self.generatePerformanceGraphs()
            updatePrompt = self.createUpdatePrompt(currentFunction, graph)
            
            proposedFunction = queryAnthropicApi(self.apiKey, self.modelName, updatePrompt)
            
            print("\nProposed Function:")
            print(proposedFunction)
            
            newFunctionCode = extractFunctionCode(proposedFunction)
            criticPrompt = self.createCriticPrompt(newFunctionCode)
            criticResponse = queryAnthropicApi(self.apiKey, self.modelName, criticPrompt)
            approved = criticResponse.strip().endswith("Decision: Yes")
            
            # Increment the update counter for this component - we count the API call even if not approved
            self.update_counts[self.targetComponent] = self.update_counts.get(self.targetComponent, 0) + 1
            
            if approved:
                # Store successful function if it's performing well
                current_performance = np.mean(list(self.rewardHistory)[-self.evaluationWindow:])
                if current_performance > self.bestPerformance:
                    self.bestPerformance = current_performance
                    self.bestRewardFunction = currentFunction
                
                self.rewardFunctionHistory.append({
                    'episode': self.episodeCount,
                    'function': newFunctionCode,
                    'performance': current_performance
                })
                
                return newFunctionCode, True
            
            # If not approved, consider reverting to best historical function
            if self.bestRewardFunction is not None:
                print("\nReverting to best historical reward function")
                return self.bestRewardFunction, True
                
            return currentFunction, False
                
        except Exception as e:
            print(f"\nError during update: {e}")
            # Still count the attempt even if there was an error
            self.update_counts[self.targetComponent] = self.update_counts.get(self.targetComponent, 0) + 1
            
            if self.bestRewardFunction is not None:
                print("\nReverting to best historical reward function due to error")
                return self.bestRewardFunction, True
            return currentFunction, False

    def logFunctionUpdate(self, component, old_func, new_func):
        """Log when a reward function is updated"""
        print(f"\nUpdating {component} reward function:")
        print("Old function:")
        print(old_func)
        print("\nNew function:")
        print(new_func)
        print("-" * 50)

    def waitingTime(self, componentName, metrics, lastUpdateEpisode):
        """Improved waiting time logic with performance checks, freezing, and update limits"""
        # Extract the component number from the component name (format: component_N)
        componentNum = int(componentName.split('_')[1]) if '_' in componentName else self.targetComponent
        
        # Check if we've reached the update limit for this component
        if self.update_counts.get(componentNum, 0) >= self.max_updates_per_run:
            # Don't even bother checking other conditions if we've hit the limit
            if metrics['currentEpisode'] % 2000 == 0:  # Only log occasionally
                print(f"\nComponent {componentNum} has reached its update limit ({self.max_updates_per_run}). No updates will be triggered.")
            return False
        
        currentEpisode = metrics['currentEpisode']
        timeSinceUpdate = currentEpisode - lastUpdateEpisode
        
        # Enforce cooldown period
        if timeSinceUpdate < self.cooldownPeriod:
            return False
            
        recentRewards = metrics['recentRewards']
        if len(recentRewards) < 100:  # Need enough history
            return False
            
        # Calculate performance metrics
        current_performance = np.mean(recentRewards[-self.evaluationWindow:])
        historical_best = np.max([np.mean(recentRewards[i:i+self.evaluationWindow]) 
                                for i in range(0, len(recentRewards)-self.evaluationWindow, self.evaluationWindow)])
        
        # Update tracking
        self.performanceHistory.append({
            'episode': currentEpisode,
            'performance': current_performance,
            'best': historical_best
        })
        
        # NEW: Performance freezing - If current performance is good, don't update
        if current_performance > 0.8 * historical_best:
            # print(f"\nMaintaining good performance for {componentName}: {current_performance:.2f} vs best {historical_best:.2f}")
            return False
        
        # Check for significant performance degradation
        if current_performance < 0.5 * historical_best:
            print(f"\nPerformance Analysis for {componentName}:")
            print(f"Current Performance: {current_performance:.2f}")
            print(f"Historical Best: {historical_best:.2f}")
            print(f"Relative Performance: {(current_performance/historical_best)*100:.1f}%")
            print(f"Update count: {self.update_counts.get(componentNum, 0)}/{self.max_updates_per_run}")
            return True
            
        return False


    def waitingTimeConstant(self, currentEpisode, interval):
        """Simple time-based waiting function that triggers at fixed intervals"""
        return currentEpisode % interval == 0 and currentEpisode > 0


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