import numpy as np
import gymnasium as gym


import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#Reward Function Specification -------------------------------------------

def angleBasedReward(observation, action):
    _, _, angle, _ = observation
    return np.cos(angle)

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





#Environment Class -------------------------------------------


class CustomCartPoleEnv(gym.Wrapper):
    def __init__(self, env, num_components=None):
        super().__init__(env)
        self.env = env
        self.rewardFunction = self.angleBasedReward
        
        # Add component tracking without breaking existing functionality
        self.using_components = bool(num_components)
        self.reward_components = {}
        if self.using_components:
            for i in range(1, num_components + 1):
                self.reward_components[f'reward_function_{i}'] = None

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        
        # Add component rewards to info without changing core functionality
        if self.using_components and any(self.reward_components.values()):
            # Calculate component rewards
            info['component_rewards'] = {}
            rewards = []
            for name, func in self.reward_components.items():
                if func and callable(func):
                    component_reward = func(observation, action)
                    rewards.append(component_reward)
                    info['component_rewards'][name] = component_reward
            reward = sum(rewards) / len(rewards) if rewards else self.rewardFunction(observation, action)
        else:
            # Original behavior
            if self.rewardFunction is None or not callable(self.rewardFunction):
                print("Warning: rewardFunction is None or not callable. Using default reward function.")
                self.rewardFunction = self.angleBasedReward
            reward = self.rewardFunction(observation, action)
            
        return observation, reward, terminated, truncated, info

    # Add new methods for component handling
    def set_component_reward(self, component_number: int, reward_function):
        """Set a specific component reward function."""
        if self.using_components:
            func_name = f'reward_function_{component_number}'
            if func_name in self.reward_components:
                self.reward_components[func_name] = reward_function
                # If it's the first component, also set it as main reward function for compatibility
                if component_number == 1:
                    self.rewardFunction = reward_function
            return True
        return False

    # Keep all existing methods unchanged
    def angleBasedReward(self, observation, action):
        _, _, angle, _ = observation
        return np.cos(angle)

    def LLMRewardFunction(self, functionString):
        # Keep existing implementation
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
        # Keep existing implementation
        self.rewardFunction = rewardFunction

    def setEnvironmentParameters(self, masscart=1.0, length=1.0, gravity=9.8):
        # Keep existing implementation
        self.env.masscart = masscart
        self.env.length = length
        self.env.gravity = gravity
        print(f"Environment parameters updated: masscart={masscart}, length={length}, gravity={gravity}")

    def updateRewardFunction(self, functionString):
        # Keep existing implementation
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

# Deep Q-Learning Agent -------------------------------------------

class DQLearningAgent:
    def __init__(self, env, stateSize, actionSize, device, learningRate=0.001, discountFactor=0.99, epsilon=1.0, epsilonDecay=0.995, epsilonMin=0.01):
        self.env = env
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.epsilonMin = epsilonMin
        self.device = device  # Add device as an attribute

        # Neural network for Q-value approximation
        self.model = self.buildModel().to(self.device)  # Move model to the specified device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate)
        self.lossFunction = torch.nn.MSELoss()

        # Replay memory
        self.memory = deque(maxlen=2000)

    def buildModel(self):
        # Define a simple neural network model for Q-value approximation
        return nn.Sequential(
            nn.Linear(self.stateSize, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.actionSize)
        )
    
    def remember(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))

    def chooseAction(self, state):
        if np.random.random() <= self.epsilon:
            # Exploration: choose a random action
            return self.env.action_space.sample()
        else:
            # Exploitation: choose the action with max Q-value for current state
            stateTensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)  # Ensure tensor is on the correct device
            with torch.no_grad():
                qValues = self.model(stateTensor)
            return torch.argmax(qValues).item()

    def replay(self, batchSize=32):
        if len(self.memory) < batchSize:
            return

        minibatch = random.sample(self.memory, batchSize)
        for state, action, reward, nextState, done in minibatch:
            # Ensure state and next state tensors are on the right device
            stateTensor = torch.FloatTensor(state).to(self.device)
            nextStateTensor = torch.FloatTensor(nextState).to(self.device)

            target = reward
            if not done:
                target = reward + self.discountFactor * torch.max(self.model(nextStateTensor)).item()

            # Clone the state tensor and get the target for the action taken
            targetTensor = self.model(stateTensor).detach().clone()
            
            # Convert the target to a torch tensor and move it to the correct device
            target = torch.tensor(target, dtype=torch.float32).to(self.device)
            targetTensor[action] = target

            # Update model
            output = self.model(stateTensor)
            targetTensor = targetTensor.to(self.device)  # Ensure targetTensor is on the right device
            loss = self.lossFunction(output, targetTensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update exploration rate
        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay


# Training Function -------------------------------------------

def trainDQLearning(agent, env, episodes=500):
    rewards = []
    for episode in range(episodes):
        observation = env.reset()[0]
        totalReward = 0
        done = False

        while not done:
            action = agent.chooseAction(observation)
            nextObservation, reward, done, _, _ = env.step(action)
            totalReward += reward
            agent.remember(observation, action, reward, nextObservation, done)
            observation = nextObservation

        rewards.append(totalReward)
        agent.replay()

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