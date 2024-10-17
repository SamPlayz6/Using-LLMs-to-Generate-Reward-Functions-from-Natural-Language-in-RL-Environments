import numpy as np
import gymnasium as gym


#Reward Function Specification -------------------------------------------

def angleBasedReward(observation, action):
    _, _, angle, _ = observation
    return np.cos(angle)

import re

def extractFunctionCode(responseString):
    startKeyword = "```python"
    endKeyword = "```"

    startIdx = responseString.find(startKeyword)
    if startIdx == -1:
        raise ValueError("No Python code block found in the response.")

    # Find the index after '```python' to extract the actual function code
    startIdx += len(startKeyword)

    # Find the end of the code block
    endIdx = responseString.find(endKeyword, startIdx)
    if endIdx == -1:
        raise ValueError("Code block not properly closed with '```'.")

    # Extract the code between the delimiters
    functionString = responseString[startIdx:endIdx].strip()

    return functionString



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





#Environment Class -------------------------------------------


class CustomCartPoleEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.rewardFunction = self.angleBasedReward

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        # Ensure rewardFunction is callable, otherwise fallback to default
        if self.rewardFunction is None or not callable(self.rewardFunction):
            print("Warning: rewardFunction is None or not callable. Using default reward function.")
            self.rewardFunction = self.angleBasedReward
        # Calculate reward
        reward = self.rewardFunction(observation, action)
        return observation, reward, terminated, truncated, info



# --


    def angleBasedReward(self, observation, action):
        _, _, angle, _ = observation
        return np.cos(angle)

    def LLMRewardFunction(self, functionString):
        localNamespace = {}
        exec(functionString, globals(), localNamespace)
        
        new_function = None
        for item in localNamespace.values():
            if callable(item):
                new_function = item
                break
        
        if new_function is None:
            raise ValueError("Invalid Function")
        
        # Set the new function as the reward function
        self.setRewardFunction(new_function)

# --

    def setRewardFunction(self, rewardFunction):
        self.rewardFunction = rewardFunction


    #dynamically update environment parameters
    def setEnvironmentParameters(self, masscart=1.0, length=1.0, gravity=9.8):
        self.env.masscart = masscart  # Example to change mass of cart
        self.env.length = length  # Example to change length of pole
        self.env.gravity = gravity  # Example to change gravity
        print(f"Environment parameters updated: masscart={masscart}, length={length}, gravity={gravity}")

        
    #update reward function dynamically
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
            # Ensure fallback to default
            self.setRewardFunction(self.angleBasedReward)


#Q-Learning Agent -------------------------------------------

class QLearningAgent:
    def __init__(self, env, n_bins=10, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.env = env
        self.n_bins = n_bins
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon


        self.bins = [
            np.linspace(-4.8, 4.8, self.n_bins),  # Position
            np.linspace(-5, 5, self.n_bins),      # Velocity
            np.linspace(-0.418, 0.418, self.n_bins),  # Angle
            np.linspace(-5, 5, self.n_bins)       # Angular velocity
        ]

        # Initialize Q-table with zeros
        self.q_table = np.zeros((self.n_bins, self.n_bins, self.n_bins, self.n_bins, env.action_space.n))

    def discretize(self, observation):
        # Discretize each dimension of the observation
        state = []
        for i, obs in enumerate(observation):
            state.append(np.digitize(obs, self.bins[i]) - 1)
        return tuple(state)



    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])










 
#Training Function -------------------------------------------

def train(agent, env, episodes=500):
    rewards = []
    for episode in range(episodes):
        observation = env.reset()[0]
        state = agent.discretize(observation)
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_observation, reward, done, _, _ = env.step(action)
            next_state = agent.discretize(next_observation)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
    
    return rewards




#API Query -------------------------------------------

import anthropic
import datetime
import json

def queryAnthropicApi(api_key, model_name, messages, max_tokens=1024):
    # Initialize the client with the given API key
    client = anthropic.Anthropic(api_key=api_key)
    
    # Generate a reward function using the provided messages
    generatedRewardFunction = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=messages
    )
    
    return generatedRewardFunction.content[0].text

def queryAnthropicExplanation(api_key, model_name, explanation_message, max_tokens=1024):
    # Initialize the client with the given API key
    client = anthropic.Anthropic(api_key=api_key)
    
    # Generate explanation for the reward function based on the provided explanation message
    explanationResponse = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=explanation_message
    )
    
    return explanationResponse.content[0].text


def logClaudeCall(rewardPrompt, rewardResponse, explanationPrompt, explanationResponse, logFile='claude_calls.jsonl'):
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'reward_function': {
            'prompt': rewardPrompt,
            'response': rewardResponse
        },
        'explanation': {
            'prompt': explanationPrompt,
            'response': explanationResponse
        }
    }
    
    with open(logFile, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')





# Environment Alterations ---------------------------------

def manualEnvironmentChange(env, episode):
    if episode % 200 == 0:  # Ask every 50 episodes
        userInput = input(f"Do you want to change the environment in episode {episode}? (y/n): ")
        if userInput.lower() == 'y':
            masscart = float(input("Enter new mass for the cart: "))
            length = float(input("Enter new length for the pole: "))
            gravity = float(input("Enter new gravity: "))
            env.setEnvironmentParameters(masscart=masscart, length=length, gravity=gravity)