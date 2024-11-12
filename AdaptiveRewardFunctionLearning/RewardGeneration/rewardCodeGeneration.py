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

def createCompositeCode():
    stabilityFunc = """
def stabilityReward(observation, action):
    x, xDot, angle, angleDot = observation
    # Focus purely on angle stability
    angle_stability = 1.0 - abs(angle) / 0.209  # 0.209 radians is about 12 degrees
    angular_velocity_component = -abs(angleDot) / 10.0  # Penalize fast angle changes
    return float(angle_stability + angular_velocity_component)
"""

    efficiencyFunc = """
def energyEfficiencyReward(observation, action):
    x, xDot, angle, angleDot = observation
    # Focus purely on minimizing movement and energy use
    cart_movement_penalty = -abs(xDot) / 5.0  # Penalize cart velocity
    angular_movement_penalty = -abs(angleDot) / 5.0  # Penalize pole angular velocity
    return float(1.0 + cart_movement_penalty + angular_movement_penalty)
"""

    timeFunc = """
def timeBasedReward(observation, action):
    x, xDot, angle, angleDot = observation
    # Simple time-based reward that encourages survival
    base_reward = 1.0
    # Add small penalties for extreme positions/angles to prevent gaming
    if abs(angle) > 0.209 or abs(x) > 2.4:  # If about to fail
        base_reward = 0.0
    return float(base_reward)
"""

    dynamicFunc = """
def dynamicRewardFunction(observation, action):
    x, xDot, angle, angleDot = observation
    
    # Get individual rewards
    stability = stabilityReward(observation, action)
    efficiency = energyEfficiencyReward(observation, action)
    timeReward = timeBasedReward(observation, action)
    
    # Combine rewards with equal weights
    return (stability + efficiency + timeReward) / 3.0
"""
    return stabilityFunc + efficiencyFunc + timeFunc + dynamicFunc



