import torch
import torch.nn as nn
from collections import deque
import numpy as np
import random

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
        self.memory = deque(maxlen=8000)

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