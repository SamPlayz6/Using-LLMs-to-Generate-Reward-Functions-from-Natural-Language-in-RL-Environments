import numpy as np
import torch
import gymnasium as gym

class EnergyBasedRewardFunction:
    def __init__(self, mass_cart=1.0, mass_pole=0.1, length=0.5, gravity=9.8):
        """
        Comprehensive energy-based reward function
        
        Parameters:
        - mass_cart: Mass of the cart
        - mass_pole: Mass of the pole
        - length: Length of the pole
        - gravity: Gravitational acceleration
        """
        self.mass_cart = mass_cart
        self.mass_pole = mass_pole
        self.length = length
        self.gravity = gravity
    
    def compute_kinetic_energy(self, state):
        """
        Full kinetic energy computation
        
        State vector: [x, x_dot, theta, theta_dot]
        """
        x_dot, theta_dot = state[1], state[3]
        
        # Cart kinetic energy
        cart_kinetic_energy = 0.5 * self.mass_cart * (x_dot**2)
        
        # Pole kinetic energy (rotational and translational)
        # Consider both rotational motion of pole and its translational component
        pole_rotational_ke = 0.5 * (self.mass_pole * (self.length**2)) * (theta_dot**2)
        
        # Translational component of pole's kinetic energy
        # Accounts for complex motion of pole tip
        pole_x_velocity = x_dot + self.length * theta_dot * np.cos(state[2])
        pole_y_velocity = self.length * theta_dot * np.sin(state[2])
        pole_translational_ke = 0.5 * self.mass_pole * (pole_x_velocity**2 + pole_y_velocity**2)
        
        return cart_kinetic_energy + pole_rotational_ke + pole_translational_ke
    
    def compute_potential_energy(self, state):
        """
        Potential energy computation
        
        Considers gravitational potential energy of the pole
        """
        # Height of pole's center of mass
        height = state[0] + self.length * np.sin(state[2])
        
        return self.mass_pole * self.gravity * height
    
    def compute_reward(self, state, action):
        """
        Reward function based on energy components with adaptive weighting
        """
        kinetic_energy = self.compute_kinetic_energy(state)
        potential_energy = self.compute_potential_energy(state)
        
        # Stability terms with length-based weighting
        angle_penalty = abs(state[2])
        velocity_penalty = state[1]**2 + state[3]**2
        
        # Dynamic weights based on pole length
        angle_weight = 0.2 * (self.length/0.5)  # Increases with pole length
        velocity_weight = 0.1 * (0.5/self.length)  # Decreases with pole length
        
        reward = -(
            kinetic_energy + 
            potential_energy + 
            angle_weight * angle_penalty + 
            velocity_weight * velocity_penalty
        )
        
        return reward
    
    def recordEpisode(self, info, steps, totalReward):
        if not hasattr(self, 'performance_history'):
            self.performance_history = []

        self.performance_history.append({
            'info': info,
            'steps': steps,
            'reward': totalReward,
            'current_length': self.length
        })

def explainability_analysis():
    """
    Demonstrate reward function components and their impact
    """
    env = gym.make('CartPole-v1')
    reward_function = EnergyBasedRewardFunction()
    
    # Collect sample states and analyze energy breakdown
    results = []
    
    for _ in range(100):
        state, _ = env.reset()
        
        # Compute energy components
        ke = reward_function.compute_kinetic_energy(state)
        pe = reward_function.compute_potential_energy(state)
        reward = reward_function.compute_reward(state, None)
        
        results.append({
            'state': state,
            'kinetic_energy': ke,
            'potential_energy': pe,
            'total_reward': reward
        })
    
    # Analysis of energy contributions
    ke_mean = np.mean([r['kinetic_energy'] for r in results])
    pe_mean = np.mean([r['potential_energy'] for r in results])
    reward_mean = np.mean([r['total_reward'] for r in results])
    
    print("Energy Contribution Analysis:")
    print(f"Mean Kinetic Energy: {ke_mean}")
    print(f"Mean Potential Energy: {pe_mean}")
    print(f"Mean Reward: {reward_mean}")
    
    return results

# Alternative Reward Function Approaches
class AlternativeRewardFunctions:
    @staticmethod
    def quaternion_stability_reward(state):
        """
        Reward based on quaternion representation of pole orientation
        """
        # Convert angle to quaternion representation
        cos_half = np.cos(state[2] / 2)
        sin_half = np.sin(state[2] / 2)
        
        # Quaternion stability metric
        quaternion_stability = 1 - abs(cos_half)
        
        return -quaternion_stability
    
    @staticmethod
    def information_theoretic_reward(state, entropy):
        """
        Reward incorporating information-theoretic principles
        """
        # Measure of state uncertainty
        state_entropy = np.sum(np.abs(state))
        
        return -(state_entropy + entropy)
    
    @staticmethod
    def model_predictive_reward(state, predicted_state, actual_state):
        """
        Reward based on prediction accuracy
        """
        prediction_error = np.mean((predicted_state - actual_state)**2)
        return -prediction_error

# Comparative Performance Analysis
def compare_reward_functions():
    """
    Compare different reward function approaches
    """
    env = gym.make('CartPole-v1')
    
    reward_approaches = {
        'Energy-Based': EnergyBasedRewardFunction().compute_reward,
        'Quaternion Stability': AlternativeRewardFunctions.quaternion_stability_reward,
        # Add more reward function approaches
    }
    
    performance_results = {}
    
    for name, reward_func in reward_approaches.items():
        # Simulate episodes with this reward function
        total_rewards = []
        for _ in range(50):  # Multiple episodes
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = env.action_space.sample()  # Random policy for comparison
                next_state, _, done, _, _ = env.step(action)
                
                # Compute reward using specific approach
                reward = reward_func(next_state, action)
                episode_reward += reward
            
            total_rewards.append(episode_reward)
        
        # Compute performance metrics
        performance_results[name] = {
            'mean_reward': np.mean(total_rewards),
            'reward_std': np.std(total_rewards)
        }
    
    # Print comparative results
    for name, metrics in performance_results.items():
        print(f"{name} Reward Approach:")
        print(f"  Mean Reward: {metrics['mean_reward']}")
        print(f"  Reward Variance: {metrics['reward_std']}")
    
    return performance_results

if __name__ == "__main__":
    # Run analyses
    explainability_analysis()
    compare_reward_functions()
