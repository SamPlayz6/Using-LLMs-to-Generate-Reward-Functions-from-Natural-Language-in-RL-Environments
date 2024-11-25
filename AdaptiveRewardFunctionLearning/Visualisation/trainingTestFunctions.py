# Cell 1: Imports, Setup, and Helper Functions
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
from pathlib import Path

# Set up plotting style
# plt.style.use('seaborn')
# sns.set_palette("husl")

current_dir = os.getcwd()  
project_root = str(Path(current_dir).parent.parent)
sys.path.append(project_root)

# Initialize environment and device
from AdaptiveRewardFunctionLearning.Prompts.prompts import device, apiKey,modelName

# Helper Functions
def runEpisode(env, agent):
    """Run a single episode and return total reward"""
    state = env.reset()[0]
    totalReward = 0
    done = False
    
    while not done:
        action = agent.chooseAction(state)
        nextState, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        totalReward += reward
        
        agent.remember(state, action, reward, nextState, done)
        state = nextState
        
    agent.replay(32)
    return totalReward

def detectJumps(rewards, windowSize=50):
    """Detect number of significant performance jumps"""
    if len(rewards) < windowSize * 2:
        return 0
    
    rollingMean = pd.Series(rewards).rolling(windowSize).mean().dropna()
    differences = rollingMean.diff()
    jumpThreshold = differences.std() * 3  # 3 sigma threshold
    
    jumps = np.where(abs(differences) > jumpThreshold)[0]
    return len(jumps)

def analyzeRewardSensibility(rewardFunction, numTests=1000):
    """Analyze if reward function outputs are sensible"""
    results = {
        "mean": 0,
        "std": 0,
        "minReward": float('inf'),
        "maxReward": float('-inf'),
        "outOfBounds": 0
    }
    
    # Generate random states within typical bounds
    states = np.random.uniform(
        low=[-2.4, -4.0, -0.209, -4.0],
        high=[2.4, 4.0, 0.209, 4.0],
        size=(numTests, 4)
    )
    
    rewards = []
    for state in states:
        reward = rewardFunction(state, 1)  # Test with both actions
        rewards.append(reward)
        
        # Update metrics
        results["minReward"] = min(results["minReward"], reward)
        results["maxReward"] = max(results["maxReward"], reward)
        
        # Check for unreasonable values
        if abs(reward) > 100:
            results["outOfBounds"] += 1
            
    results["mean"] = np.mean(rewards)
    results["std"] = np.std(rewards)
    
    return results

def performUpdate(env, updateSystem, episode):
    """Perform reward function update"""
    print(f"\nAttempting update at episode {episode}")
    currentFunc = env.rewardFunction
    newFunction, updated = updateSystem.validateAndUpdate(currentFunc)
    
    if updated:
        print("Reward function updated")
        env.setRewardFunction(newFunction)
        updateSystem.lastUpdateEpisode = episode
        return True 
    return False   



from AdaptiveRewardFunctionLearning.RewardGeneration.rewardCritic import RewardUpdateSystem
from AdaptiveRewardFunctionLearning.RewardGeneration.rewardCodeGeneration import analyseFailure, updateComponentWeight

def updateCompositeRewardFunction(env, updateSystem, metrics, dynamicRewardFunction):
    """
    Updates composite reward function components based on waiting time and performance
    
    Args:
        env: The environment instance
        updateSystem: System handling LLM updates
        metrics: Dictionary containing current performance metrics
        dynamicRewardFunction: The dynamic reward function being used
    """

    if not hasattr(dynamicRewardFunction, 'weights'):
        dynamicRewardFunction.weights = {
        'stability': {'value': 0.33, 'lastUpdate': 0},
        'efficiency': {'value': 0.33, 'lastUpdate': 0},
        'time': {'value': 0.34, 'lastUpdate': 0}
    }
    dynamicRewardFunction.lastObservation = None
    dynamicRewardFunction.episodeEnded = False
    
    # print(f"\nChecking composite components for updates at episode {metrics['currentEpisode']}")
    
    # Check each composite component
    for componentName in ['stability', 'efficiency', 'time']:
        # Skip if component doesn't exist in weights
        if not hasattr(dynamicRewardFunction, 'weights') or componentName not in dynamicRewardFunction.weights:
            continue

        # print("Makes it through weights existing 1")
            
        # Check if this component should be updated
        if updateSystem.waitingTime(componentName, metrics, 
                      dynamicRewardFunction.weights[componentName]['lastUpdate'],
                      threshold=0.5):
            
            print(f"\nAttempting to update composite component: {componentName}")
            
            # Analyze failure type for weight adjustment
            failureType = analyseFailure(dynamicRewardFunction.lastObservation)
            
            # Update component weight
            oldWeight = dynamicRewardFunction.weights[componentName]['value']
            updateComponentWeight(componentName, failureType)
            newWeight = dynamicRewardFunction.weights[componentName]['value']
            
            # Log the update
            if not hasattr(dynamicRewardFunction, 'compositeHistory'):
                dynamicRewardFunction.compositeHistory = []
            
            dynamicRewardFunction.compositeHistory.append({
                'component': componentName,
                'episode': metrics['currentEpisode'],
                'oldWeight': oldWeight,
                'newWeight': newWeight,
                'failureType': failureType,
                'metrics': metrics.copy()
            })
            
            # Update last update time
            dynamicRewardFunction.weights[componentName]['lastUpdate'] = metrics['currentEpisode']
            
            print(f"Updated {componentName} weight: {oldWeight:.3f} -> {newWeight:.3f}")

from datetime import datetime


# Helper function to save plots
def savePlot(fig, experimentName, subFolder, configIdx=None, plotType="results"):

    # Create logs directory with subfolder if it doesn't exist
    logs_dir = Path(current_dir).parent.parent / 'logs' / subFolder
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    
    # Create filename
    if configIdx is not None:
        filename = f"{experimentName}_config{configIdx}_{plotType}_{timestamp}.png"
    else:
        filename = f"{experimentName}_{plotType}_{timestamp}.png"
    
    # Full path for saving
    filepath = os.path.join(logs_dir, filename)
    
    # Save figure
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Saved plot: {filename} in {subFolder}")



def plotExperimentResults(rewards, metrics, title):
    """Plot comprehensive experiment results"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot rewards
    ax1.plot(rewards, alpha=0.6, label='Episode Reward')
    ax1.plot(pd.Series(rewards).rolling(50).mean(), 
             label='50-Episode Moving Average', linewidth=2)
    ax1.set_title(f'{title} - Rewards Over Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True)
    
    # Plot performance jumps
    episodes = list(metrics.keys())
    jumps = [m['jumps'] for m in metrics.values()]
    ax2.bar(episodes, jumps, alpha=0.7)
    ax2.set_title('Performance Jumps Detected')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Number of Jumps')
    ax2.grid(True)
    
    # Plot reward sensibility metrics
    avgRewards = [m['averageReward'] for m in metrics.values()]
    ax3.plot(episodes, avgRewards, marker='o')
    ax3.set_title('Average Reward per 100 Episodes')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average Reward')
    ax3.grid(True)
    
    plt.tight_layout()
    savePlot(fig, title.replace(" ", ""))
    
    return fig
