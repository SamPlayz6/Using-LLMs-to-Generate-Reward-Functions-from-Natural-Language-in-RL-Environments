def movingAverageAndStd(data, windowSize=100):
    average = np.convolve(data, np.ones(windowSize) / windowSize, mode='valid')
    std = [np.std(data[i:i+windowSize]) for i in range(len(data) - windowSize + 1)]
    return average, std


plt.figure(figsize=(15, 10))


#Total Reward Comparison
plt.subplot(2, 2, 1)
for label, rewards in rewardsDict.items():
    plt.plot(np.arange(episodes), rewards, label=label)


plt.xlabel("Episodes")
plt.ylabel("Total Reward")

plt.legend()

# 2.Running Average of Rewards

# avgBaseline, stdBaseline = movingAverageAndStd(baselineRewards, windowSize=100)
# avgLLM1, stdLLM1 = movingAverageAndStd(LLM1Rewards, windowSize=100)
# avgLLM2, stdLLM2 = movingAverageAndStd(LLM2Rewards, windowSize=100)

plt.figure(figsize=(10, 6))


for label, rewards in rewardsDict.items():
    avg, std = movingAverageAndStd(rewards, windowSize=50)
    plt.plot(np.arange(len(avg)), avg, label=label)
    plt.fill_between(np.arange(len(avg)), avg - std, avg + std, alpha=0.2, label=f"{label} Variance")

    # print("------------------")
    # print(avg)
    # print(std)





plt.xlabel("Episodes")
plt.ylabel("Average Reward")


plt.xlim(0, episodes)
plt.legend()
plt.tight_layout()
plt.show()

# 3. Episode Duration
def episodeDuration(env, rewardModel, episodes=500):
    durations = []
    for episode in range(episodes):
        observation = env.reset()
        done = False
        duration = 0

        while not done:
            action = env.action_space.sample()
            observation, reward, done, _, _ = env.step(action)
            duration += 1  
        durations.append(duration)

    return durations

# Get the episode durations
durationsDict = {label: episodeDuration(env, reward_fn, episodes) for label, reward_fn in rewardFunctions}


# Plot Episode Duration
plt.subplot(2, 2, 3)
for label, durations in durationsDict.items():
    plt.plot(np.arange(episodes), durations, label=label)


plt.xlabel("Episodes")
plt.ylabel("Episode Duration(Steps)")
plt.legend()

# 4.action distribution
# def getActionDistributions(env, rewardModels, episodes):
#     actionDistributions = {}
#     for rewardModelName, rewardModel in rewardModels:
#         actions = []
#         for episode in range(episodes):
#             observation = env.reset()[0]
#             done = False

#             while not done:
#                 action = env.action_space.sample() 
#                 actions.append(action)
#                 observation, _, done, _, _ = env.step(action)

#         actionDistributions[rewardModelName] = actions 

#     return actionDistributions



# actionDistributions = getActionDistributions(env, rewardFunctions, episodes)

# # Plot action distributions
# plt.figure(figsize=(10, 6))
# for rewardModelName, actions in actionDistributions.items():
#     plt.hist(actions, bins=2, alpha=0.7, label=f"{rewardModelName} Actions")

# plt.xlabel("Actions (0=Left, 1=Right)")
# plt.ylabel("Frequency")
# plt.legend()
# plt.show()

# 5. Reward Distribution
plt.figure(figsize=(15, 5))

# Plot for Baseline Rewards
for i, (label, rewards) in enumerate(rewardsDict.items(), 1):
    plt.subplot(1, len(rewardsDict.items()), i)
    plt.hist(rewards, bins=20, alpha=0.7)
    plt.title(label)
    plt.xlabel("Reward")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()