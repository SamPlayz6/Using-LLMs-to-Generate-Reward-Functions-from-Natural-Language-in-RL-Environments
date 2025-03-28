from collections import deque
import numpy as np


def trainDQLearning(agent, env, numEpisodes, updateSystem=None, onEpisodeEnd=None):
    rewards = []
    episodeLengths = []
    
    # Track average rewards for stabilizing exploration rate
    recent_rewards = deque(maxlen=100)
    best_reward = float('-inf')
    best_reward_epoch = 0
    
    for episode in range(numEpisodes):
        observation = env.reset()[0]
        totalReward = 0
        steps = 0
        done = False
        
        while not done:
            action = agent.chooseAction(observation)
            nextObservation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Clip the reward to avoid extreme values that might destabilize learning
            clipped_reward = max(-10, min(10, reward))
            
            totalReward += reward  # Keep original reward for reporting
            steps += 1
            
            # Store clipped reward in memory
            agent.remember(observation, action, clipped_reward, nextObservation, done)
            observation = nextObservation
            
            # Train less frequently with a larger sample when memory is substantial
            # Early in training, train more often with smaller batches
            if len(agent.memory) > agent.batchSize:
                if len(agent.memory) < 10000 and steps % 2 == 0:
                    # Early training: more frequent, smaller batches
                    agent.replay(forcedBatchSize=32)
                elif steps % 8 == 0:  # Less frequent updates (8 steps vs 4)
                    # Later training: less frequent, larger batches
                    agent.replay()
        
        rewards.append(totalReward)
        episodeLengths.append(steps)
        recent_rewards.append(totalReward)
        
        # Track best performance and adjust exploration if we're plateauing
        avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
        
        # If we're setting a new high score, remember when it happened
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_reward_epoch = episode
        
        # If we haven't improved in a long time, inject some exploration
        if episode - best_reward_epoch > 500 and episode % 100 == 0:
            # Temporarily increase exploration to escape local optimum
            agent.epsilon = min(0.3, agent.epsilon * 1.5)
            print(f"Episode {episode}: Performance plateau detected. Increasing exploration to {agent.epsilon:.4f}")
            best_reward_epoch = episode  # Reset counter
        
        if onEpisodeEnd:
            onEpisodeEnd(env, updateSystem, episode, totalReward, steps)
        
        # Ensure multiple training updates at the end of each episode to maximize learning
        if len(agent.memory) > agent.batchSize:
            # Multiple training iterations after each episode for better learning
            for _ in range(min(4, steps // 50)):  # More updates for longer episodes
                agent.replay()
    
    return agent, env, rewards

