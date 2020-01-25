import gym
env = gym.make('BipedalWalker-v2')
env.reset()
score = 0

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        score+=reward
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()
print(score)
