import re

def extractFunctionCode(responseString):
    # Updated pattern to match from the first 'def' to the end of the string
    function_pattern = r"(def\s+dynamicRewardFunction\(.*\):[\s\S]*)"
    match = re.search(function_pattern, responseString)

    if not match:
        raise ValueError("No valid function definition found in the response.")

    functionString = match.group(1)
    return functionString.strip()



class RewardUpdateSystem:
    def __init__(self, apiKey: str, modelName: str, maxHistoryLength: int = 100, target_component: int = 1):
        self.apiKey = apiKey
        self.modelName = modelName
        self.rewardHistory = deque(maxlen=maxHistoryLength)
        self.episodeCount = 0
        self.lastUpdateEpisode = 0
        self.target_component = target_component
    
    def generateRewardGraph(self):
        """Generate a base64 encoded string of the reward history plot."""
        plt.figure(figsize=(10, 6))
        plt.plot(list(self.rewardHistory), label=f'Reward Component {self.target_component}')
        plt.title(f'Component {self.target_component} Reward History')
        plt.xlabel('Steps')
        plt.ylabel('Reward Value')
        plt.grid(True)
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode()

    def createUpdatePrompt(self, currentFunction: str, graph: str):
        return [
            {
                "role": "user",
                "content": f"""Analyze this reward component function and its performance graph, then suggest specific modifications.
                
                Current Function:
                {currentFunction}
                
                Performance Graph (base64 encoded):
                {graph}
                
                Requirements:
                1. The function MUST be named 'reward_function_{self.target_component}'
                2. Focus on stability (pole angle and angular velocity)
                3. Include detailed inline comments explaining each change
                4. Compare old vs new values in comments
                
                Output only the modified function with detailed inline comments explaining changes."""
            }
        ]

    def createCriticPrompt(self, proposedFunction: str):
        return [
            {
                "role": "user",
                "content": f"""Act as a reward function critic. Analyze this proposed reward component:

                {proposedFunction}

                Evaluate:
                1. Is the function properly named 'reward_function_{self.target_component}'?
                2. Are reward calculations mathematically sound?
                3. Does it focus appropriately on its specific aspect?
                4. Are there any potential issues?
                
                End your response with EXACTLY one line containing only "Decision: Yes" or "Decision: No"."""
            }
        ]
        
    def recordReward(self, info: dict):
        """Record the specific component reward value."""
        if 'component_rewards' in info:
            component_reward = info['component_rewards'].get(f'reward_function_{self.target_component}')
            if component_reward is not None:
                self.rewardHistory.append(component_reward)
        
    def validateAndUpdate(self, currentFunction: str):
        """Generate and validate updated reward function with delays between API calls."""
        try:
            # First API call for function update
            print(f"\nGenerating new reward function for component {self.target_component}...")
            graph = self.generateRewardGraph()
            updatePrompt = self.createUpdatePrompt(currentFunction, graph)
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
                return proposedFunction, True
            return currentFunction, False
            
        except Exception as e:
            print(f"\nError during update: {e}")
            return currentFunction, False

    def waitingTime(self, episode: int):
        """Determine if it's time to update the reward function."""
        UPDATE_INTERVAL = 100
        return episode - self.lastUpdateEpisode >= UPDATE_INTERVAL
