import torch
import numpy as np
import gymnasium as gym

class ContinuousCartPoleEnv:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.state_dim = 4  # (x, ẋ, θ, θ̇)
        self.action_dim = 1  # Continuous force application
    
    def reset(self):
        """
        Reset environment and return initial state tensor
        
        Returns:
            torch.Tensor: Initial state representation
        """
        state, _ = self.env.reset()
        return torch.FloatTensor(state)
    
    def step(self, action):
        """
        Apply continuous action to environment
        
        Args:
            action (torch.Tensor): Continuous force to apply
        
        Returns:
            Tuple of (state, reward, done, info)
        """
        # Clip action to reasonable force range
        force = np.clip(action.item(), -10, 10)
        
        # Step environment
        next_state, reward, done, truncated, _ = self.env.step(
            0 if force < 0 else 1  # Discrete mapping of continuous force
        )
        
        return (
            torch.FloatTensor(next_state),
            torch.tensor(reward, dtype=torch.float32),
            done or truncated
        )

class StochasticOptimalControlPolicy(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # Mean network for action
        self.mean_network = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_dim)
        )
        
        # Learned log standard deviation
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        """
        Compute stochastic policy distribution
        
        Args:
            state (torch.Tensor): System state
        
        Returns:
            torch.distributions.Normal: Action distribution
        """
        mean = self.mean_network(state)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mean, std)
    
    def sample_action(self, state):
        """
        Sample action from policy distribution
        
        Returns:
            torch.Tensor: Sampled continuous action
        """
        dist = self(state)
        action = dist.rsample()  # Differentiable sampling
        return action

class TensorTrajectory:
    def __init__(self):
        """
        Structured trajectory representation
        
        Stores tensors for:
        - States: (x, ẋ, θ, θ̇)
        - Actions: Continuous forces
        - Rewards: Scalar rewards
        - Log probabilities: Action distribution log probs
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []
    
    def add(self, state, action, reward, log_prob, done):
        """
        Add step to trajectory
        
        Args:
            state (torch.Tensor): System state
            action (torch.Tensor): Applied action
            reward (torch.Tensor): Received reward
            log_prob (torch.Tensor): Action log probability
            done (bool): Episode termination flag
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_returns(self, gamma=0.99):
        """
        Compute discounted returns
        
        Args:
            gamma (float): Discount factor
        
        Returns:
            torch.Tensor: Discounted cumulative rewards
        """
        returns = []
        R = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            R = reward + gamma * R * (1.0 - done)
            returns.insert(0, R)
        return torch.tensor(returns)

def meta_learning_stochastic_control():
    """
    Meta-learning framework bridging stochastic optimal control
    and reinforcement learning
    """
    env = ContinuousCartPoleEnv()
    policy = StochasticOptimalControlPolicy(
        state_dim=env.state_dim, 
        action_dim=env.action_dim
    )
    
    # Optimization setup
    optimizer = torch.optim.Adam(policy.parameters())
    
    # Training loop
    for episode in range(1000):
        trajectory = TensorTrajectory()
        state = env.reset()
        
        while True:
            # Sample action from policy
            action_dist = policy(state)
            action = action_dist.rsample()
            log_prob = action_dist.log_prob(action)
            
            # Environment step
            next_state, reward, done = env.step(action)
            
            # Store trajectory
            trajectory.add(state, action, reward, log_prob, done)
            
            state = next_state
            
            if done:
                break
        
        # Compute returns and loss
        returns = trajectory.compute_returns()
        loss = -torch.mean(
            torch.stack(trajectory.log_probs) * returns
        )
        
        # Update policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Execute meta-learning
if __name__ == "__main__":
    meta_learning_stochastic_control()
