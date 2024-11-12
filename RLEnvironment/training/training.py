from collections import deque
import numpy as np

def trainDQLearning(agent, env, numEpisodes, updateSystem=None, onEpisodeEnd=None):
    rewards = []
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

        if updateSystem:
            updateSystem.recordEpisode(info, steps, totalReward)

        rewards.append(totalReward)

        if onEpisodeEnd:
            onEpisodeEnd(env, updateSystem, episode)

        agent.replay(32)

    return agent, env, rewards