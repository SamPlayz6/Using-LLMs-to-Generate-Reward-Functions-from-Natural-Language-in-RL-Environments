import datetime
import json
from collections import deque



# In rewardCodeGeneration.py

def createDynamicFunctions():
    stabilityFunc = """
def stabilityReward(observation, action):
    x, xDot, angle, angleDot = observation
    
    # Primary component: angle-based reward (higher when pole is upright)
    angle_reward = 1.0 - (abs(angle) / 0.209)  # Normalize to [0, 1]
    
    # Secondary component: angular velocity penalty (smaller is better)
    velocity_penalty = min(0.5, abs(angleDot) / 8.0)  # Cap at 0.5
    
    # Combine components
    return float(angle_reward - velocity_penalty)
"""

    efficiencyFunc = """
def energyEfficiencyReward(observation, action):
    x, xDot, angle, angleDot = observation
    
    # Primary component: position-based reward (higher when cart is centered)
    position_reward = 1.0 - (abs(x) / 2.4)  # Normalize to [0, 1]
    
    # Secondary component: velocity penalty (smaller is better)
    velocity_penalty = min(0.5, abs(xDot) / 5.0)  # Cap at 0.5
    
    # Combine components
    return float(position_reward - velocity_penalty)
"""
    return {
        'stability': stabilityFunc,
        'efficiency': efficiencyFunc
    }

    

# Initialize the base functions by executing their strings
initial_funcs = {}
namespace = {}
function_defs = createDynamicFunctions()
exec(function_defs['stability'], namespace)
exec(function_defs['efficiency'], namespace)

# Export the initialized functions
stabilityReward = namespace['stabilityReward']
efficiencyReward = namespace['energyEfficiencyReward']

def dynamicRewardFunction(observation, action, metrics=None):
    """
    Dynamic reward function with balanced weighting of stability and efficiency components.
    Keeps track of observation history and includes safety mechanisms to prevent
    weight imbalances that could cause performance drops.
    """
    # TEMPORARY MODIFICATION: Initialize weights with constant 0.5/0.5 values
    if not hasattr(dynamicRewardFunction, 'weights'):
        dynamicRewardFunction.weights = {
            'stability': {'value': 0.5, 'lastUpdate': 0},  # Fixed at 0.5
            'efficiency': {'value': 0.5, 'lastUpdate': 0}  # Fixed at 0.5
        }
        # Initialize tracking systems
        dynamicRewardFunction.function_updates = {}
        dynamicRewardFunction.lastObservation = observation
        dynamicRewardFunction.episodeEnded = False
        dynamicRewardFunction.performance_tracking = {
            'recent_rewards': deque(maxlen=100),
            'best_performance': 0,
            'last_weight_change_episode': 0
        }
    
    # Store the observation for failure analysis
    dynamicRewardFunction.lastObservation = observation
    
    # Get base functions or their updated versions
    functionStrings = createDynamicFunctions()
    
    # Override with any updated functions
    if hasattr(dynamicRewardFunction, 'function_updates'):
        for func_name, new_code in dynamicRewardFunction.function_updates.items():
            if func_name.startswith('rewardFunction1'):
                functionStrings['stability'] = new_code
            elif func_name.startswith('rewardFunction2'):
                functionStrings['efficiency'] = new_code
    
    # Create namespace and execute the functions
    namespace = {}
    exec(functionStrings['stability'], namespace)
    exec(functionStrings['efficiency'], namespace)
    
    # Get rewards using potentially updated functions - trap exceptions
    try:
        stability = namespace['stabilityReward'](observation, action)
    except Exception as e:
        print(f"Error in stability reward: {e}")
        stability = 0  # Fallback on error
        
    try:
        efficiency = namespace['efficiencyReward'](observation, action)
    except Exception as e:
        print(f"Error in efficiency reward: {e}")
        efficiency = 0  # Fallback on error
    
    # SAFETY BOUNDS: Limit extreme reward values to prevent instability
    stability = max(-10, min(10, stability))  # Cap stability reward
    efficiency = max(-10, min(10, efficiency))  # Cap efficiency reward
    
    # TEMPORARY MODIFICATION: Keep weights constant at 0.5 and 0.5 
    # instead of using dynamic weights from dynamicRewardFunction.weights
    stability_weight = 0.5
    efficiency_weight = 0.5
    
    # Update weights dictionary for consistency (though these values won't be used in calculation)
    dynamicRewardFunction.weights['stability']['value'] = 0.5
    dynamicRewardFunction.weights['efficiency']['value'] = 0.5
    
    # Calculate final weighted reward
    reward = (stability * stability_weight + efficiency * efficiency_weight)
    
    # Track reward for performance monitoring
    if hasattr(dynamicRewardFunction, 'performance_tracking'):
        dynamicRewardFunction.performance_tracking['recent_rewards'].append(reward)
    
    return reward

def analyseFailure(lastObservation):
    if lastObservation is None:
        return 'timeout'
    x, xDot, angle, angleDot = lastObservation
    
    if abs(x) > 2.4:  # Position failure
        return 'position'
    elif abs(angle) > 0.209:  # Angle failure
        return 'angle'
    elif abs(xDot) > 3.0:  # Velocity failure
        return 'velocity'
    return 'timeout'

def adjustWeightsAfterEpisode(weights, failureType):
    # TEMPORARY MODIFICATION: Return fixed weights (0.5, 0.5) without any adjustments
    # This function is temporarily disabled to maintain constant weights
    
    # Create a new dictionary with fixed weights
    fixed_weights = {
        'stability': 0.5,
        'efficiency': 0.5
    }
    
    # Track episodes for consistency but don't use for adjustments
    if not hasattr(adjustWeightsAfterEpisode, 'current_episode'):
        adjustWeightsAfterEpisode.current_episode = 0
    adjustWeightsAfterEpisode.current_episode += 1
    
    # Return the fixed weights
    return fixed_weights

def updateComponentWeight(component, failureType):
    """
    TEMPORARY MODIFICATION: Function disabled to maintain constant weights (0.5, 0.5)
    Original purpose: Update component weights based on failure types.
    """
    # Track episodes for consistency but don't actually update weights
    if not hasattr(updateComponentWeight, 'current_episode'):
        updateComponentWeight.current_episode = 0
    updateComponentWeight.current_episode += 1
    
    # Store weight history for consistency
    if not hasattr(dynamicRewardFunction, 'weight_history'):
        dynamicRewardFunction.weight_history = []
    
    # Fix weights at 0.5 directly in the dynamicRewardFunction
    dynamicRewardFunction.weights[component]['value'] = 0.5
    
    # Log that an update was attempted but weights remained fixed
    dynamicRewardFunction.weight_history.append({
        'episode': updateComponentWeight.current_episode,
        'component': component,
        'old_value': 0.5,
        'new_value': 0.5,
        'failure_type': failureType,
        'note': 'Weight updates temporarily disabled'
    })