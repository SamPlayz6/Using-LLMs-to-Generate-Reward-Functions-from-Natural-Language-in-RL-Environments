import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F

class MetaRewardLearner:
    def __init__(self, state_dim, action_dim, meta_learning_rate=0.01, inner_lr=0.001):
        """
        Meta-learning framework for reward function optimization
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            meta_learning_rate (float): Learning rate for meta-optimization
            inner_lr (float): Learning rate for inner loop policy optimization
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.meta_learning_rate = meta_learning_rate
        self.inner_lr = inner_lr
        
        # Parameterized reward function
        self.reward_params = nn.Parameter(torch.zeros(3))
        
        # Policy network
        self.policy_network = PolicyNetwork(state_dim, action_dim)
        
        # Optimizers
        self.reward_optimizer = optim.Adam([self.reward_params], lr=meta_learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=inner_lr)
    
    def parameterized_reward(self, state, action):
        """
        Flexible reward function with learnable parameters
        
        Params:
        - params[0]: Potential energy weight
        - params[1]: Kinetic energy weight
        - params[2]: Angle penalty weight
        """
        potential_energy = self.reward_params[0] * state[1] * 9.8
        kinetic_energy = self.reward_params[1] * 0.5 * (state[3]**2)
        angle_penalty = self.reward_params[2] * abs(state[2])
        
        return -(potential_energy + kinetic_energy + angle_penalty)
    
    def compute_meta_loss(self, trajectories):
        """
        Compute meta-loss using multiple formulations
        
        1. Performance Variance Minimization
        2. Entropy-Regularized Performance
        3. Multi-Task Performance Consistency
        """
        # Collect performance metrics across trajectories
        performances = []
        entropies = []
        
        for trajectory in trajectories:
            # Extract states, actions, rewards
            states = torch.tensor(trajectory['states'], dtype=torch.float32)
            actions = torch.tensor(trajectory['actions'], dtype=torch.float32)
            
            # Compute trajectory performance (total reward)
            trajectory_performance = sum(trajectory['rewards'])
            performances.append(trajectory_performance)
            
            # Compute action distribution entropy
            action_dist = self.policy_network(states)
            entropy = action_dist.entropy().mean()
            entropies.append(entropy)
        
        # Performance Variance Minimization
        performance_variance = torch.var(torch.tensor(performances))
        
        # Entropy-Regularized Performance
        mean_entropy = torch.mean(torch.tensor(entropies))
        
        # Multi-Task Performance Consistency
        performance_std = torch.std(torch.tensor(performances))
        
        # Combine meta-loss components
        meta_loss = (
            performance_variance +  # Minimize performance variance
            -mean_entropy +  # Maximize action entropy
            performance_std  # Penalize inconsistent performance
        )
        
        return meta_loss
    
    def meta_update(self, trajectories):
        """
        Meta-learning update step
        """
        # Compute meta-loss
        meta_loss = self.compute_meta_loss(trajectories)
        
        # Zero gradients and compute gradients
        self.reward_optimizer.zero_grad()
        meta_loss.backward()
        
        # Update reward function parameters
        self.reward_optimizer.step()
        
        return meta_loss.item()

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        """
        Compute action distribution
        
        Returns:
            Normal distribution over actions
        """
        mean = self.network(state)
        std = torch.exp(self.log_std)
        return Normal(mean, std)
    
    def select_action(self, state):
        """
        Sample an action from the policy
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        dist = self(state)
        action = dist.sample()
        return action.numpy()[0]

def run_episode(env, policy_network, meta_reward_learner, render=False):
    """
    Run a single episode with meta-learned reward function
    
    Returns:
        Dictionary containing trajectory information
    """
    state, _ = env.reset()
    done = False
    truncated = False
    
    trajectory = {
        'states': [],
        'actions': [],
        'rewards': [],
        'next_states': []
    }
    
    total_reward = 0
    
    while not (done or truncated):
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state)
        
        # Select action
        action = policy_network.select_action(state)
        
        # Step environment
        next_state, reward, done, truncated, _ = env.step(action)
        
        # Compute meta-learned reward
        meta_reward = meta_reward_learner.parameterized_reward(next_state, action)
        
        # Store trajectory
        trajectory['states'].append(state_tensor)
        trajectory['actions'].append(torch.FloatTensor([action]))
        trajectory['rewards'].append(meta_reward)
        trajectory['next_states'].append(torch.FloatTensor(next_state))
        
        # Update state and total reward
        state = next_state
        total_reward += meta_reward
        
        if render:
            env.render()
    
    return trajectory

def meta_learning_cartpole():
    # Initialize environment
    env = gym.make('CartPole-v1')
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize meta-reward learner
    meta_learner = MetaRewardLearner(state_dim, action_dim)
    
    # Meta-training loop
    num_meta_iterations = 1000
    episodes_per_iteration = 10
    
    for meta_iter in range(num_meta_iterations):
        # Collect trajectories
        trajectories = []
        for _ in range(episodes_per_iteration):
            trajectory = run_episode(env, meta_learner.policy_network, meta_learner)
            trajectories.append(trajectory)
        
        # Perform meta-update
        meta_loss = meta_learner.meta_update(trajectories)
        
        # Periodic evaluation
        if meta_iter % 50 == 0:
            print(f"Meta-Iteration {meta_iter}, Meta-Loss: {meta_loss}")
            
            # Render a sample episode
            run_episode(env, meta_learner.policy_network, meta_learner, render=True)
    
    return meta_learner

# Main execution
if __name__ == "__main__":
    meta_learner = meta_learning_cartpole()
