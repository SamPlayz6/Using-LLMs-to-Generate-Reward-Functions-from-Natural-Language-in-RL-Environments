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
            
        rewards.append(totalReward)
        episodeLengths.append(steps)
        
        # Only use onEpisodeEnd for all update logic
        if onEpisodeEnd:
            onEpisodeEnd(env, updateSystem, episode, totalReward, steps)
        
        # Regular training step
        if len(agent.memory) > 32:
            agent.replay(32)
        
        # Periodic logging
        if episode % 100 == 0:
            avgReward = np.mean(rewards[-100:]) if rewards else 0
            avgSteps = np.mean(episodeLengths[-100:]) if episodeLengths else 0
            print(f"\nEpisode {episode}/{numEpisodes}")
            print(f"Average Reward: {avgReward:.2f}")
            print(f"Average Steps: {avgSteps:.2f}")
    
    return agent, env, rewards