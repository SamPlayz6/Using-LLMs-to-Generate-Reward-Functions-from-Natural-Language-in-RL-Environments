from collections import deque
import numpy as np

def trainDQLearning(agent, env, updateSystem, num_episodes, on_episode_end=None):
    rewards = []
    for episode in range(num_episodes):
        observation = env.reset()[0]
        total_reward = 0
        done = False

        while not done:
            action = agent.chooseAction(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            agent.remember(observation, action, reward, next_observation, done)
            observation = next_observation

        rewards.append(total_reward)

        if on_episode_end:
            on_episode_end(env, updateSystem, episode)

        agent.replay(32)

    return agent, env, rewards