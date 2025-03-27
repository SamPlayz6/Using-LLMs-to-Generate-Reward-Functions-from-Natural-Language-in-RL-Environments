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
    # Initialize weights if not exist - use more balanced weights with higher stability
    if not hasattr(dynamicRewardFunction, 'weights'):
        dynamicRewardFunction.weights = {
            'stability': {'value': 0.55, 'lastUpdate': 0},  # Slightly higher stability
            'efficiency': {'value': 0.45, 'lastUpdate': 0}  # Slightly lower efficiency
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
    
    # GET CURRENT WEIGHTS with safety bounds enforcement
    stability_weight = dynamicRewardFunction.weights['stability']['value']
    efficiency_weight = dynamicRewardFunction.weights['efficiency']['value']
    
    # SAFETY CHECK: Make sure weights are within acceptable ranges
    # This helps prevent weight drift from normalization or other issues
    stability_min, stability_max = 0.4, 0.7
    efficiency_min, efficiency_max = 0.3, 0.6
    
    # Apply safety bounds to ensure stability remains dominant enough
    stability_weight = max(stability_min, min(stability_max, stability_weight))
    efficiency_weight = max(efficiency_min, min(efficiency_max, efficiency_weight))
    
    # Re-normalize weights after bounding
    total = stability_weight + efficiency_weight
    if total > 0:  # Avoid division by zero
        stability_weight /= total
        efficiency_weight /= total
        
    # Update weights with bounded values
    dynamicRewardFunction.weights['stability']['value'] = stability_weight
    dynamicRewardFunction.weights['efficiency']['value'] = efficiency_weight
    
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
    # DRAMATICALLY REDUCED weight adjustments for stability
    # Changed from 0.1-0.2 to 0.01-0.02 (10x smaller)
    weightChanges = {
        'stability': 0.0,
        'efficiency': 0.0
    }
    
    # Use tiny incremental changes instead of large adjustments
    if failureType == 'angle':  # Failed due to angle > 12 degrees
        weightChanges['stability'] = +0.02  # Was 0.2
        weightChanges['efficiency'] = -0.02  # Was -0.2
        
    elif failureType == 'position':  # Failed due to cart position
        weightChanges['stability'] = +0.01  # Was 0.1
        weightChanges['efficiency'] = -0.01  # Was -0.1
        
    elif failureType == 'velocity':  # Failed due to excessive speed
        weightChanges['efficiency'] = +0.02  # Was 0.2
        weightChanges['stability'] = -0.02  # Was -0.2
        
    elif failureType == 'timeout':  # Succeeded until max steps
        # Make success-based adjustments even smaller to prevent over-optimization
        weightChanges['efficiency'] = +0.005  # Was 0.1
        weightChanges['stability'] = -0.005  # Was -0.1
    
    # TIGHTER BOUNDS: Keep stability higher with narrower range
    # Changed from [0.2, 0.8] to [0.4, 0.7] for stability
    # Changed from [0.2, 0.8] to [0.3, 0.6] for efficiency
    
    # Apply changes with tighter bounds for better stability
    stability_min, stability_max = 0.4, 0.7  # More restrictive range for stability
    efficiency_min, efficiency_max = 0.3, 0.6  # More restrictive range for efficiency
    
    # Apply bounds specific to each component
    if 'stability' in weights:
        weights['stability'] = max(stability_min, min(stability_max, weights['stability'] + weightChanges['stability']))
    if 'efficiency' in weights:
        weights['efficiency'] = max(efficiency_min, min(efficiency_max, weights['efficiency'] + weightChanges['efficiency']))
    
    # COOLDOWN TRACKING: Add cooldown to prevent rapid oscillations
    # Track last update episode for each component
    if not hasattr(adjustWeightsAfterEpisode, 'last_update_episode'):
        adjustWeightsAfterEpisode.last_update_episode = 0
        
    # LIMIT FREQUENCY: Only allow weight changes every 100 episodes
    current_episode = getattr(adjustWeightsAfterEpisode, 'current_episode', 0) + 1
    adjustWeightsAfterEpisode.current_episode = current_episode
    
    # Skip adjustments if we're updating too frequently
    if current_episode - adjustWeightsAfterEpisode.last_update_episode < 100:
        # Return without changes if we're in cooldown period
        return weights
        
    # Update the last update episode
    adjustWeightsAfterEpisode.last_update_episode = current_episode
    
    # Normalize weights to ensure they sum to 1
    total = sum(weights.values())
    for key in weights:
        weights[key] /= total
    
    return weights

def updateComponentWeight(component, failureType):
    """
    Update component weights with much smaller increments and strict cooldown periods
    to prevent excessive shifts and performance drops.
    """
    # Initialize cooldown tracking if not exists
    if not hasattr(updateComponentWeight, 'last_update_episode'):
        updateComponentWeight.last_update_episode = {}
    if not hasattr(updateComponentWeight, 'current_episode'):
        updateComponentWeight.current_episode = 0
        
    # Increment episode counter
    updateComponentWeight.current_episode += 1
    
    # Initialize component's last update if not tracked
    if component not in updateComponentWeight.last_update_episode:
        updateComponentWeight.last_update_episode[component] = 0
        
    # ENFORCE COOLDOWN: Only allow updates every 200 episodes per component
    cooldown_period = 200
    if updateComponentWeight.current_episode - updateComponentWeight.last_update_episode.get(component, 0) < cooldown_period:
        # Skip update during cooldown
        return
        
    # Store current weights for comparison and possible rollback
    if not hasattr(dynamicRewardFunction, 'previous_weights'):
        dynamicRewardFunction.previous_weights = {}
        
    # Save previous weight before changes
    dynamicRewardFunction.previous_weights[component] = dynamicRewardFunction.weights[component]['value']
    
    # TINY ADJUSTMENTS: Reduced from 0.2 to 0.02 (10x smaller)
    adjustment_size = 0.02  # Was 0.2
    
    # Only make appropriate adjustments based on failure type
    if component == 'stability':
        if failureType == 'angle':
            # Increase stability weight slightly when angle failures occur
            dynamicRewardFunction.weights[component]['value'] += adjustment_size
        elif failureType == 'timeout':
            # Slightly decrease stability on success (much smaller adjustment)
            dynamicRewardFunction.weights[component]['value'] -= adjustment_size * 0.25  # Only 0.005
    elif component == 'efficiency':
        if failureType == 'velocity':
            # Increase efficiency slightly when velocity failures occur
            dynamicRewardFunction.weights[component]['value'] += adjustment_size
        elif failureType == 'timeout':
            # Slightly increase efficiency on success (much smaller adjustment)
            dynamicRewardFunction.weights[component]['value'] += adjustment_size * 0.25  # Only 0.005
            
    # ENFORCE BOUNDS: Tighter ranges for each component
    # Stability should be kept relatively high (0.4-0.7)
    # Efficiency should be kept moderate (0.3-0.6)
    component_bounds = {
        'stability': (0.4, 0.7),   # Min/max for stability
        'efficiency': (0.3, 0.6),  # Min/max for efficiency
    }
    
    # Apply bounds if defined for this component
    if component in component_bounds:
        min_val, max_val = component_bounds[component]
        dynamicRewardFunction.weights[component]['value'] = max(min_val, min(max_val, 
                                                    dynamicRewardFunction.weights[component]['value']))
    
    # SAFETY CHECK: Ensure weight doesn't change by more than the adjustment_size
    max_change = adjustment_size
    if abs(dynamicRewardFunction.weights[component]['value'] - dynamicRewardFunction.previous_weights[component]) > max_change:
        # Limit change to max allowed
        if dynamicRewardFunction.weights[component]['value'] > dynamicRewardFunction.previous_weights[component]:
            dynamicRewardFunction.weights[component]['value'] = dynamicRewardFunction.previous_weights[component] + max_change
        else:
            dynamicRewardFunction.weights[component]['value'] = dynamicRewardFunction.previous_weights[component] - max_change
            
    # Update cooldown timestamp
    updateComponentWeight.last_update_episode[component] = updateComponentWeight.current_episode
            
    # Normalize weights after update (ensure they sum to 1)
    total = sum(w['value'] for w in dynamicRewardFunction.weights.values())
    for w in dynamicRewardFunction.weights.values():
        w['value'] /= total
        
    # Log the weight change
    if not hasattr(dynamicRewardFunction, 'weight_history'):
        dynamicRewardFunction.weight_history = []
        
    dynamicRewardFunction.weight_history.append({
        'episode': updateComponentWeight.current_episode,
        'component': component,
        'old_value': dynamicRewardFunction.previous_weights[component],
        'new_value': dynamicRewardFunction.weights[component]['value'],
        'failure_type': failureType
    })