import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from typing import Dict, List, Callable
import optuna

class RewardFunctionMetaLearner:
    def __init__(self, state_dim, action_dim):
        """
        Meta-learning framework for reward function optimization
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Parameterized reward function generator
        self.reward_function_network = self.create_reward_network()
        
        # Hyperparameter search space
        self.hyperparameter_space = {
            'learning_rate': (1e-4, 1e-2),
            'energy_weights': {
                'kinetic': (-1, 1),
                'potential': (-1, 1),
                'stability': (-1, 1)
            },
            'regularization': {
                'l1_strength': (0, 1e-3),
                'l2_strength': (0, 1e-3)
            }
        }
    
    def create_reward_network(self):
        """
        Create a flexible neural network for reward function generation
        
        Returns:
            nn.Module: Reward function generator
        """
        return nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single output for reward
        )
    
    def generate_reward_function(self, sample_params=None):
        """
        Generate a parameterized reward function
        
        Args:
            sample_params (dict): Optional parameter sampling
        
        Returns:
            Callable: Reward computation function
        """
        if sample_params is None:
            sample_params = self.sample_hyperparameters()
        
        def parameterized_reward(state, action):
            """
            Flexible reward computation
            
            Args:
                state (np.ndarray): System state
                action (np.ndarray): Taken action
            
            Returns:
                float: Computed reward
            """
            # Neural network based reward computation
            state_tensor = torch.FloatTensor(state)
            base_reward = self.reward_function_network(state_tensor).item()
            
            # Energy-based components
            kinetic_contrib = sample_params['energy_weights']['kinetic'] * np.sum(state[1::2]**2)
            potential_contrib = sample_params['energy_weights']['potential'] * np.abs(state[2])
            stability_contrib = sample_params['energy_weights']['stability'] * np.abs(state[3])
            
            # Regularization terms
            l1_penalty = sample_params['regularization']['l1_strength'] * np.sum(np.abs(state))
            l2_penalty = sample_params['regularization']['l2_strength'] * np.sum(state**2)
            
            # Combined reward
            total_reward = (
                base_reward + 
                kinetic_contrib + 
                potential_contrib + 
                stability_contrib - 
                l1_penalty - 
                l2_penalty
            )
            
            return total_reward
        
        return parameterized_reward
    
    def sample_hyperparameters(self):
        """
        Sample hyperparameters from predefined space
        
        Returns:
            dict: Sampled hyperparameters
        """
        return {
            'learning_rate': np.random.uniform(
                self.hyperparameter_space['learning_rate'][0],
                self.hyperparameter_space['learning_rate'][1]
            ),
            'energy_weights': {
                k: np.random.uniform(v[0], v[1]) 
                for k, v in self.hyperparameter_space['energy_weights'].items()
            },
            'regularization': {
                k: np.random.uniform(v[0], v[1])
                for k, v in self.hyperparameter_space['regularization'].items()
            }
        }
    
    def meta_optimize(self, num_iterations=100, episodes_per_iteration=10):
        """
        Meta-optimization of reward function
        
        Args:
            num_iterations (int): Number of meta-learning iterations
            episodes_per_iteration (int): Episodes to evaluate each reward function
        
        Returns:
            List of performance metrics
        """
        # Performance tracking
        performance_history = []
        
        for iteration in range(num_iterations):
            # Generate reward function variants
            reward_functions = [
                self.generate_reward_function() 
                for _ in range(10)
            ]
            
            # Evaluate each reward function variant
            iteration_performances = []
            
            for reward_func in reward_functions:
                # Simulate performance
                performance = self.evaluate_reward_function(
                    reward_func, 
                    episodes=episodes_per_iteration
                )
                iteration_performances.append(performance)
            
            # Select and update based on performance
            best_performance = max(iteration_performances)
            performance_history.append(best_performance)
            
            # Adaptive learning rate adjustment
            self.update_reward_network(best_performance)
        
        return performance_history
    
    def evaluate_reward_function(self, reward_func, episodes=10):
        """
        Evaluate a specific reward function
        
        Args:
            reward_func (Callable): Reward computation function
            episodes (int): Number of episodes to simulate
        
        Returns:
            float: Performance metric
        """
        env = gym.make('CartPole-v1')
        total_rewards = []
        
        for _ in range(episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = env.action_space.sample()
                next_state, _, done, _, _ = env.step(action)
                
                # Compute reward using generated function
                reward = reward_func(next_state, action)
                episode_reward += reward
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)
    
    def update_reward_network(self, performance):
        """
        Update reward function generator based on performance

        Args:
            performance (float): Performance metric
        """
        # Use absolute value of performance to ensure positive learning rate
        abs_performance = abs(performance)

        # Simple performance-based update with guaranteed positive learning rate
        optimizer = optim.Adam(
            self.reward_function_network.parameters(),
            lr=0.001 * abs_performance
        )

        # Simulate a gradient update
        optimizer.zero_grad()
        loss = torch.tensor(-performance, requires_grad=True)
        loss.backward()
        optimizer.step()

        
    def recordEpisode(self, info, steps, totalReward):
        if not hasattr(self, 'episode_history'):
            self.episode_history = []

        self.episode_history.append({
            'info': info,
            'steps': steps,
            'reward': totalReward
        })
        
        
    def meta_update(self, trajectories):
        """
        Update the reward function based on recent trajectories

        Args:
            trajectories (list): List of recent trajectories

        Returns:
            float: Loss value from the update
        """
        # Extract performance from trajectories
        performance = np.mean([np.mean(traj['rewards']) for traj in trajectories])

        # Update the reward network
        self.update_reward_network(performance)

        return float(performance)
        
        
# Bayesian Hyperparameter Optimization
class HyperparameterTuner:
    def __init__(self, meta_learner):
        """
        Advanced hyperparameter optimization
        
        Args:
            meta_learner (RewardFunctionMetaLearner): Meta-learning framework
        """
        self.meta_learner = meta_learner
    
    def objective(self, trial):
        """
        Optuna objective function for hyperparameter optimization
        
        Args:
            trial (optuna.Trial): Optimization trial
        
        Returns:
            float: Performance metric
        """
        # Sample hyperparameters
        learning_rate = trial.suggest_loguniform(
            'learning_rate', 
            1e-4, 1e-2
        )
        
        kinetic_weight = trial.suggest_uniform(
            'kinetic_weight', 
            -1, 1
        )
        
        potential_weight = trial.suggest_uniform(
            'potential_weight', 
            -1, 1
        )
        
        # Create custom reward function with sampled parameters
        custom_params = {
            'energy_weights': {
                'kinetic': kinetic_weight,
                'potential': potential_weight,
                'stability': trial.suggest_uniform('stability_weight', -1, 1)
            },
            'regularization': {
                'l1_strength': trial.suggest_loguniform('l1_strength', 1e-5, 1e-2),
                'l2_strength': trial.suggest_loguniform('l2_strength', 1e-5, 1e-2)
            },
            'learning_rate': learning_rate
        }
        
        # Generate and evaluate reward function
        reward_func = self.meta_learner.generate_reward_function(custom_params)
        performance = self.meta_learner.evaluate_reward_function(reward_func)
        
        return performance
    
    def optimize(self, n_trials=100):
        """
        Run Bayesian optimization
        
        Args:
            n_trials (int): Number of optimization trials
        
        Returns:
            Dict: Best hyperparameters and performance
        """
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value
        }

def main():
    # Initialize meta-learning framework
    env = gym.make('CartPole-v1')
    meta_learner = RewardFunctionMetaLearner(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    
    # Perform meta-optimization
    performance_history = meta_learner.meta_optimize()
    
    # Hyperparameter tuning
    tuner = HyperparameterTuner(meta_learner)
    best_configuration = tuner.optimize()
    
    print("Meta-Learning Performance History:", performance_history)
    print("Best Hyperparameter Configuration:", best_configuration)

if __name__ == "__main__":
    main()
