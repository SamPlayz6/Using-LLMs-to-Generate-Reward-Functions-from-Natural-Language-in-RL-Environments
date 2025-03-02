from collections import deque
import numpy as np


def trainDQLearning(agent, env, numEpisodes, updateSystem=None, onEpisodeEnd=None):
    rewards = []
    episodeLengths = []
    
    for episode in range(numEpisodes):
        observation = env.reset()[0]
        totalReward = 0
        steps = 0
        done = False
        
        while not done:
            action = agent.chooseAction(observation)
            nextObservation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            totalReward += reward
            steps += 1
            
            agent.remember(observation, action, reward, nextObservation, done)
            observation = nextObservation
            
            # Train less frequently - every 4 steps if enough samples
            if len(agent.memory) > agent.batchSize and steps % 4 == 0:
                agent.replay()
        
        rewards.append(totalReward)
        episodeLengths.append(steps)
        
        if onEpisodeEnd:
            onEpisodeEnd(env, updateSystem, episode, totalReward, steps)
        
        # Ensure at least one training update per episode
        if len(agent.memory) > agent.batchSize:
            agent.replay()
    
    return agent, env, rewards

