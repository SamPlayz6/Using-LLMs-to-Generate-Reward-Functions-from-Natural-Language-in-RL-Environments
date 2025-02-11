import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
from collections import deque
import random

class RewardNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        input_dim = state_dim + action_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Better initialization
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, state_action):
        return self.network(state_action)

class RewardFunctionMetaLearner:  # Changed class name back to match existing imports
    def __init__(self, state_dim, action_dim, meta_learning_rate=0.001, inner_lr=0.001):
        """
        Enhanced meta-learning framework for reward function optimization
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            meta_learning_rate (float): Learning rate for meta-optimization
            inner_lr (float): Learning rate for inner loop policy optimization
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_value = 1.0
        
        # Initialize networks
        self.reward_network = RewardNetwork(state_dim, action_dim)
        self.policy_network = PolicyNetwork(state_dim, action_dim)
        
        # Initialize optimizers with better parameters
        self.reward_optimizer = optim.Adam(self.reward_network.parameters(), 
                                         lr=meta_learning_rate,
                                         weight_decay=0.01)  # L2 regularization
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), 
                                         lr=inner_lr)
        
        # Add learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.reward_optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=10000)
        
        # Track best performance for early stopping
        self.best_performance = float('-inf')
        self.best_state_dict = None

    def generate_reward_function(self):
        """Method required for compatibility with 3_performance.ipynb"""
        return self.parameterized_reward
    
    def process_state(self, state):
        """Normalize state components"""
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        elif not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        # Normalize state components
        normalized_state = torch.tensor([
            state[0] / 2.4,     # Cart position normalized by track width
            state[1] / 10.0,    # Velocity normalized by typical range
            state[2] / 0.209,   # Angle normalized by allowed range
            state[3] / 10.0     # Angular velocity normalized
        ])
        
        return normalized_state
    
    def parameterized_reward(self, state, action):
        """
        Enhanced reward function using neural network
        """
        try:
            # Process state
            state = self.process_state(state)
            
            # Process action - handle all possible input types
            if isinstance(action, (int, np.int64)):
                action = torch.tensor([float(action)], dtype=torch.float32)
            elif isinstance(action, float):
                action = torch.tensor([action], dtype=torch.float32)
            elif isinstance(action, np.ndarray):
                action = torch.from_numpy(action.astype(np.float32))
            elif isinstance(action, torch.Tensor):
                action = action.float()
            else:
                raise ValueError(f"Unexpected action type: {type(action)}")
                
            # Ensure action is 1D tensor
            if action.dim() == 0:
                action = action.unsqueeze(0)
            
            # Combine state and action
            state_action = torch.cat([state, action])
            
            # Compute reward
            with torch.no_grad():
                reward = self.reward_network(state_action)
            
            return float(reward.item())
            
        except Exception as e:
            print(f"Error in parameterized_reward: {str(e)}")
            print(f"State type: {type(state)}, Action type: {type(action)}")
            print(f"State shape: {state.shape if hasattr(state, 'shape') else 'no shape'}")
            print(f"Action shape: {action.shape if hasattr(action, 'shape') else 'no shape'}")
            raise e
    
    def compute_meta_loss(self, trajectories):
        """
        Enhanced meta-loss computation with multiple components
        """
        total_loss = 0
        batch_size = len(trajectories)
        
        for trajectory in trajectories:
            # Extract data
            states = torch.stack([self.process_state(s) for s in trajectory['states']])
            next_states = torch.stack([self.process_state(s) for s in trajectory['next_states']])
            actions = torch.tensor(trajectory['actions'])
            
            # Compute current and next values
            current_state_action = torch.cat([states, actions], dim=1)
            next_state_action = torch.cat([next_states, actions], dim=1)
            
            current_values = self.reward_network(current_state_action)
            next_values = self.reward_network(next_state_action)
            
            # TD error
            td_error = (next_values - current_values).pow(2).mean()
            
            # Stability term (penalize extreme angles and positions)
            stability_loss = (torch.abs(states[:, 2]).mean() +  # angle
                            torch.abs(states[:, 0]).mean())     # position
            
            # Combine losses
            trajectory_loss = td_error + 0.1 * stability_loss
            
            total_loss += trajectory_loss
        
        return total_loss / batch_size
    
    def process_trajectory(self, trajectory):
        """Process trajectory for replay buffer"""
        processed_data = []
        
        try:
            states = trajectory['states']
            next_states = trajectory['next_states']
            actions = trajectory['actions']
            rewards = trajectory.get('rewards', [0] * len(states))  # Default to 0 if not provided
            
            for i in range(len(states)):
                if isinstance(states[i], tuple):  # Handle if states are still in memory tuples
                    processed_data.append({
                        'state': states[i][0],  # Assuming state is first element
                        'next_state': states[i][3],  # Assuming next_state is fourth element
                        'action': states[i][1],  # Assuming action is second element
                        'reward': rewards[i]
                    })
                else:
                    processed_data.append({
                        'state': states[i],
                        'next_state': next_states[i],
                        'action': actions[i],
                        'reward': rewards[i]
                    })
                    
        except Exception as e:
            print(f"Error processing trajectory: {str(e)}")
            print(f"Trajectory keys: {trajectory.keys()}")
            print(f"States type: {type(states)}")
            if len(states) > 0:
                print(f"First state type: {type(states[0])}")
            raise e
        
        return processed_data
    
    def meta_update(self, trajectories):
        """
        Enhanced meta-learning update with experience replay and multiple updates
        """
        # Add new trajectories to replay buffer
        for trajectory in trajectories:
            self.replay_buffer.extend(self.process_trajectory(trajectory))
        
        total_loss = 0
        num_updates = 5
        
        for _ in range(num_updates):
            # Sample batch
            batch_size = min(128, len(self.replay_buffer))
            batch = random.sample(list(self.replay_buffer), batch_size)
            
            # Create trajectory-like structure for batch
            batch_trajectory = {
                'states': [b['state'] for b in batch],
                'next_states': [b['next_state'] for b in batch],
                'actions': [b['action'] for b in batch]
            }
            
            # Compute loss
            meta_loss = self.compute_meta_loss([batch_trajectory])
            
            # Add L2 regularization
            l2_reg = sum(p.pow(2).sum() for p in self.reward_network.parameters())
            loss = meta_loss + 0.01 * l2_reg
            
            # Update with gradient clipping
            self.reward_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_network.parameters(), self.clip_value)
            self.reward_optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_updates
        
        # Update learning rate
        self.scheduler.step(avg_loss)
        
        # Update best model if performance improved
        if -avg_loss > self.best_performance:
            self.best_performance = -avg_loss
            self.best_state_dict = self.reward_network.state_dict()
        
        return avg_loss
        
    def recordEpisode(self, info, steps, totalReward):
        """Record episode information for tracking performance"""
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
            print(f"\nMeta-Learning Metrics at episode {len(self.episode_history)}:")
            print(f"Recent average reward: {np.mean([e['reward'] for e in self.episode_history[-100:]]):.2f}")
            print(f"Recent average steps: {np.mean([e['steps'] for e in self.episode_history[-100:]]):.2f}")

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
        
        # Better initialization
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, state):
        """Compute action distribution"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        mean = self.network(state)
        std = torch.exp(self.log_std)
        return Normal(mean, std)
    
    def select_action(self, state):
        """Sample an action from the policy"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        dist = self(state)
        action = dist.sample()
        return action.numpy()[0]

def run_episode(env, policy_network, meta_reward_learner, render=False):
    """Run a single episode with meta-learned reward function"""
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
        # Select action
        action = policy_network.select_action(state)
        
        # Step environment
        next_state, reward, done, truncated, _ = env.step(action)
        
        # Compute meta-learned reward
        meta_reward = meta_reward_learner.parameterized_reward(next_state, action)
        
        # Store trajectory
        trajectory['states'].append(state)
        trajectory['actions'].append(action)
        trajectory['rewards'].append(meta_reward)
        trajectory['next_states'].append(next_state)
        
        # Update state and total reward
        state = next_state
        total_reward += meta_reward
        
        if render:
            env.render()
    
    return trajectory

def meta_learning_cartpole():
    """Main meta-learning training loop with early stopping"""
    # Initialize environment
    env = gym.make('CartPole-v1')
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize meta-reward learner
    meta_learner = RewardFunctionMetaLearner(state_dim, action_dim)
    
    # Training parameters
    num_meta_iterations = 1000
    episodes_per_iteration = 10
    patience = 10
    no_improve_count = 0
    best_performance = float('-inf')
    
    for meta_iter in range(num_meta_iterations):
        # Collect trajectories
        trajectories = []
        total_episode_reward = 0
        
        for _ in range(episodes_per_iteration):
            trajectory = run_episode(env, meta_learner.policy_network, meta_learner)
            trajectories.append(trajectory)
            total_episode_reward += sum(trajectory['rewards'])
        
        # Compute average episode reward
        avg_episode_reward = total_episode_reward / episodes_per_iteration
        
        # Perform meta-update
        meta_loss = meta_learner.meta_update(trajectories)
        
        # Early stopping check
        if avg_episode_reward > best_performance:
            best_performance = avg_episode_reward
            no_improve_count = 0
            # Save best model
            meta_learner.best_state_dict = meta_learner.reward_network.state_dict()
        else:
            no_improve_count += 1
        
        # Early stopping
        if no_improve_count >= patience:
            print(f"Early stopping triggered at iteration {meta_iter}")
            # Restore best model
            meta_learner.reward_network.load_state_dict(meta_learner.best_state_dict)
            break
        
        # Periodic evaluation
        if meta_iter % 50 == 0:
            print(f"Meta-Iteration {meta_iter}, Meta-Loss: {meta_loss}, Avg Reward: {avg_episode_reward}")
    
    return meta_learner

# For compatibility with 3_performance.ipynb
def parameterized_reward(observation, action):
    """Wrapper function for compatibility"""
    if not hasattr(parameterized_reward, 'learner'):
        env = gym.make('CartPole-v1')
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        parameterized_reward.learner = RewardFunctionMetaLearner(state_dim, action_dim)
    
    return parameterized_reward.learner.parameterized_reward(observation, action)

# Main execution
if __name__ == "__main__":
    meta_learner = meta_learning_cartpole()