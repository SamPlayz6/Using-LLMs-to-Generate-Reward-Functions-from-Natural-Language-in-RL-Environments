import torch
import torch.nn as nn
from collections import deque
import numpy as np
import random

import torch
import torch.nn as nn
from collections import deque
import numpy as np
import random

class DQLearningAgent:
    def __init__(self, env, stateSize, actionSize, device, 
                 learningRate=0.0005,  # Reduced from 0.001
                 discountFactor=0.99, 
                 epsilon=1.0, 
                 epsilonDecay=0.999,  # Slower decay
                 epsilonMin=0.01,
                 replayBufferSize=100000,
                 batchSize=32,  # Back to original
                 targetUpdateFreq=200):  # Much less frequent updates
        
        self.env = env
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.epsilonMin = epsilonMin
        self.device = device
        self.batchSize = batchSize
        self.targetUpdateFreq = targetUpdateFreq
        self.trainingSteps = 0  # Counter for training steps

        # Main network for Q-value approximation
        self.model = self.buildModel().to(self.device)
        
        # Target network for stable learning
        self.targetModel = self.buildModel().to(self.device)
        self.targetModel.load_state_dict(self.model.state_dict())
        self.targetModel.eval()  # Set to evaluation mode as it's not directly trained

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate)
        self.lossFunction = torch.nn.MSELoss()

        # Larger replay memory
        self.memory = deque(maxlen=replayBufferSize)

    def buildModel(self):
        # Slightly wider network to handle more complex patterns
        return nn.Sequential(
            nn.Linear(self.stateSize, 128),  # Increased from 24
            nn.ReLU(),
            nn.Linear(128, 128),  # Added second hidden layer
            nn.ReLU(),
            nn.Linear(128, self.actionSize)
        )
    
    def remember(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))

    def chooseAction(self, state):
        if np.random.random() <= self.epsilon:
            return self.env.action_space.sample()
        
        stateTensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qValues = self.model(stateTensor)
        return torch.argmax(qValues).item()

    def replay(self, forcedBatchSize=None):
        batchSize = forcedBatchSize if forcedBatchSize else self.batchSize
        
        if len(self.memory) < batchSize:
            return

        # Train only once per replay call instead of multiple times
        minibatch = random.sample(self.memory, batchSize)
        
        # Prepare batch tensors
        states = torch.tensor([t[0] for t in minibatch], dtype=torch.float32).to(self.device)
        actions = torch.tensor([t[1] for t in minibatch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([t[2] for t in minibatch], dtype=torch.float32).to(self.device)
        nextStates = torch.tensor([t[3] for t in minibatch], dtype=torch.float32).to(self.device)
        dones = torch.tensor([t[4] for t in minibatch], dtype=torch.float32).to(self.device)

        # Current Q values
        currentQValues = self.model(states)
        currentQValues = currentQValues.gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            nextQValues = self.targetModel(nextStates)
            maxNextQ = nextQValues.max(1)[0]
            targetQValues = rewards + (1 - dones) * self.discountFactor * maxNextQ

        # Update model
        loss = self.lossFunction(currentQValues.squeeze(), targetQValues)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.trainingSteps += 1

        # Update target network less frequently
        if self.trainingSteps % self.targetUpdateFreq == 0:
            self.targetModel.load_state_dict(self.model.state_dict())

        # Update exploration rate
        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay


