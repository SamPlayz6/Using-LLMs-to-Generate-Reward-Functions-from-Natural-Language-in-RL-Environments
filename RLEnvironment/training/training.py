from collections import deque
import numpy as np

def trainDQLearning(agent, env, episodes=500):
    rewards = []
    for episode in range(episodes):
        observation = env.reset()[0]
        totalReward = 0
        done = False

        while not done:
            action = agent.chooseAction(observation)
            nextObservation, reward, done, _, _ = env.step(action)
            totalReward += reward
            agent.remember(observation, action, reward, nextObservation, done)
            observation = nextObservation

        rewards.append(totalReward)
        agent.replay()

    return rewards