import datetime
import json



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
    # Initialize weights if not exist
    if not hasattr(dynamicRewardFunction, 'weights'):
        dynamicRewardFunction.weights = {
            'stability': {'value': 0.6, 'lastUpdate': 0},
            'efficiency': {'value': 0.4, 'lastUpdate': 0}
        }
        dynamicRewardFunction.function_updates = {}
        dynamicRewardFunction.lastObservation = observation
        dynamicRewardFunction.episodeEnded = False
    
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
    
    # Get rewards using potentially updated functions
    stability = namespace['stabilityReward'](observation, action)
    efficiency = namespace['efficiencyReward'](observation, action)
    
    # Combine rewards with current weights
    reward = (stability * dynamicRewardFunction.weights['stability']['value'] + 
             efficiency * dynamicRewardFunction.weights['efficiency']['value'])
    
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
    weightChanges = {
        'stability': 0.0,
        'efficiency': 0.0
    }
    
    # More aggressive weight adjustments for faster adaptation
    if failureType == 'angle':  # Failed due to angle > 12 degrees
        weightChanges['stability'] = +0.2
        weightChanges['efficiency'] = -0.2
        
    elif failureType == 'position':  # Failed due to cart position
        weightChanges['stability'] = +0.1
        weightChanges['efficiency'] = -0.1
        
    elif failureType == 'velocity':  # Failed due to excessive speed
        weightChanges['efficiency'] = +0.2
        weightChanges['stability'] = -0.2
        
    elif failureType == 'timeout':  # Succeeded until max steps
        weightChanges['efficiency'] = +0.1
        weightChanges['stability'] = -0.1
    
    # Apply changes with bounds (prevent any weight from going below 0.2 or above 0.8)
    for key in weights:
        weights[key] = max(0.2, min(0.8, weights[key] + weightChanges[key]))
    
    # Normalize weights to ensure they sum to 1
    total = sum(weights.values())
    for key in weights:
        weights[key] /= total
    
    return weights

def updateComponentWeight(component, failureType):
    # Update logic specific to each component
    if component == 'stability':
        if failureType == 'angle':
            dynamicRewardFunction.weights[component]['value'] += 0.2
    elif component == 'efficiency':
        if failureType == 'velocity':
            dynamicRewardFunction.weights[component]['value'] += 0.2
            
    # Normalize weights after update
    total = sum(w['value'] for w in dynamicRewardFunction.weights.values())
    for w in dynamicRewardFunction.weights.values():
        w['value'] /= total