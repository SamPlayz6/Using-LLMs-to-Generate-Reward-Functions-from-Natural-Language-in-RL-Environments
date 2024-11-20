import anthropic
import datetime
import json

def queryAnthropicApi(api_key, model_name, messages, max_tokens=1024):
    # Initialize the client with the given API key
    client = anthropic.Anthropic(api_key=api_key)
    
    # Generate a reward function using the provided messages
    generatedRewardFunction = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=messages
    )
    
    return generatedRewardFunction.content[0].text

def queryAnthropicExplanation(api_key, model_name, explanation_message, max_tokens=1024):
    # Initialize the client with the given API key
    client = anthropic.Anthropic(api_key=api_key)
    
    # Generate explanation for the reward function based on the provided explanation message
    explanationResponse = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=explanation_message
    )
    
    return explanationResponse.content[0].text



# Composite Function Details

def createDynamicFunctions():
    stabilityFunc = """
def stabilityReward(observation, action):
    x, xDot, angle, angleDot = observation
    # Focus purely on angle stability
    angle_stability = 1.0 - abs(angle) / 0.209
    angular_velocity_component = -abs(angleDot) / 10.0
    return float(angle_stability + angular_velocity_component)
"""

    efficiencyFunc = """
def energyEfficiencyReward(observation, action):
    x, xDot, angle, angleDot = observation
    cart_movement_penalty = -abs(xDot) / 5.0
    angular_movement_penalty = -abs(angleDot) / 5.0
    return float(1.0 + cart_movement_penalty + angular_movement_penalty)
"""

    timeFunc = """
def timeBasedReward(observation, action):
    x, xDot, angle, angleDot = observation
    base_reward = 1.0
    if abs(angle) > 0.209 or abs(x) > 2.4:
        base_reward = 0.0
    return float(base_reward)
"""
    return {
        'stability': stabilityFunc,
        'efficiency': efficiencyFunc,
        'time': timeFunc
    }


from AdaptiveRewardFunctionLearning.RewardGeneration.rewardCritic import RewardUpdateSystem

def dynamicRewardFunction(observation, action, metrics=None):
    # Initialize weights and tracking if not exist
    if not hasattr(dynamicRewardFunction, 'weights'):
        dynamicRewardFunction.weights = {
            'stability': {'value': 0.33, 'lastUpdate': 0},
            'efficiency': {'value': 0.33, 'lastUpdate': 0},
            'time': {'value': 0.34, 'lastUpdate': 0}
        }
        dynamicRewardFunction.lastObservation = observation
        dynamicRewardFunction.episodeEnded = False
    
    # Update last observation
    dynamicRewardFunction.lastObservation = observation
    
    # Get the function strings from createDynamicFunctions
    functionStrings = createDynamicFunctions()
    
    # Create namespace and execute the string functions
    namespace = {}
    exec(functionStrings['stability'], namespace)
    exec(functionStrings['efficiency'], namespace)
    exec(functionStrings['time'], namespace)
    
    # Get individual rewards using the executed functions
    stability = namespace['stabilityReward'](observation, action)
    efficiency = namespace['energyEfficiencyReward'](observation, action)
    timeReward = namespace['timeBasedReward'](observation, action)
    
    # Combine rewards with current weights
    return (stability * dynamicRewardFunction.weights['stability']['value'] + 
            efficiency * dynamicRewardFunction.weights['efficiency']['value'] + 
            timeReward * dynamicRewardFunction.weights['time']['value'])





def analyseFailure(lastObservation):
    if lastObservation is None:
        return 'timeout'  # or some default failure type
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
        'efficiency': 0.0,
        'time': 0.0
    }
    
    # Adjust based on failure type
    if failureType == 'angle':  # Failed due to angle > 12 degrees (0.209 radians)
        # Increase stability weight, slightly decrease others
        weightChanges['stability'] = +0.15  # Significant increase for angle stability
        weightChanges['efficiency'] = -0.1  # Reduce efficiency priority
        weightChanges['time'] = -0.05      # Small reduction in time priority
        
    elif failureType == 'position':  # Failed due to cart position > 2.4
        # Need more stability and efficiency
        weightChanges['stability'] = +0.1   # More stability helps position indirectly
        weightChanges['efficiency'] = +0.1  # More efficiency reduces wild movements
        weightChanges['time'] = -0.2       # Reduce time priority significantly
        
    elif failureType == 'velocity':  # Failed due to too much speed
        # Need more efficiency focus
        weightChanges['efficiency'] = +0.15  # Significant increase for movement efficiency
        weightChanges['stability'] = -0.1    # Reduce stability priority
        weightChanges['time'] = -0.05       # Small reduction in time priority
        
    elif failureType == 'timeout':  # Succeeded until max steps
        # Reward successful behavior
        weightChanges['time'] = +0.1        # Increase time weight as it's doing well
        weightChanges['efficiency'] = +0.05  # Slight increase in efficiency
        weightChanges['stability'] = -0.15   # Can reduce stability focus
    
    # Apply changes with bounds (prevent any weight from going too high or low)
    for key in weights:
        weights[key] = max(0.1, min(0.8, weights[key] + weightChanges[key]))
    
    # Normalize weights to ensure they sum to 1
    total = sum(weights.values())
    for key in weights:
        weights[key] /= total
    
    return weights


def updateComponentWeight(component, failureType):
    # Update logic specific to each component
    if component == 'stability':
        if failureType == 'angle':
            dynamicRewardFunction.weights[component]['value'] += 0.1
    elif component == 'efficiency':
        if failureType == 'velocity':
            dynamicRewardFunction.weights[component]['value'] += 0.1
    elif component == 'time':
        if failureType == 'timeout':
            dynamicRewardFunction.weights[component]['value'] += 0.1
            
    # Normalize weights after update
    total = sum(w['value'] for w in dynamicRewardFunction.weights.values())
    for w in dynamicRewardFunction.weights.values():
        w['value'] /= total



