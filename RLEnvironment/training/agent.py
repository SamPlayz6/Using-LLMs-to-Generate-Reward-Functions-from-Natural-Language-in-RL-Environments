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
                 learningRate=0.0003,  # Further reduced learning rate for stability
                 discountFactor=0.99, 
                 epsilon=1.0, 
                 epsilonDecay=0.9995,  # Even slower decay to prevent premature exploitation
                 epsilonMin=0.05,  # Higher minimum exploration rate
                 replayBufferSize=100000,
                 batchSize=64,  # Larger batch size for more stable updates
                 targetUpdateFreq=500):  # Less frequent target updates for stability
        
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
        # Stronger gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        self.trainingSteps += 1

        # Update target network less frequently for stability
        if self.trainingSteps % self.targetUpdateFreq == 0:
            # Use soft update instead of hard update for more stability
            with torch.no_grad():
                for target_param, param in zip(self.targetModel.parameters(), self.model.parameters()):
                    target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)

        # More conservative epsilon decay, slow down as epsilon gets smaller
        if self.epsilon > self.epsilonMin:
            # Decay more slowly as we approach the minimum
            decay_factor = self.epsilonDecay + 0.0001 * (1 - self.epsilon/1.0)
            self.epsilon *= decay_factor
            self.epsilon = max(self.epsilonMin, self.epsilon)  # Ensure we don't go below minimum


