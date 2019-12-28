import gym
env = gym.make('Assault-ram-v0')
env = env.unwrapped
env.reset()
score = 0

action_size = env.action_space.n
state_size = env.observation_space.shape[0]

print("Action size", action_size, "state size", state_size)

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
