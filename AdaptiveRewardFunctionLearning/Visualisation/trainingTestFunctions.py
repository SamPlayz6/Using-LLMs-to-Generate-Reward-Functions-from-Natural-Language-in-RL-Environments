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

from datetime import datetime


# Helper function to save plots
def savePlot(fig, experimentName, configIdx=None, plotType="results"):
    # Create logs directory if it doesn't exist
    logs_dir = Path(current_dir).parent.parent / 'logs/RobustnessResults'
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
    print(f"Saved plot: {filename}")


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
