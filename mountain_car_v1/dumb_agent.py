import gym
import random
env = gym.make('MountainCar-v0')
env.reset()
score = 0

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = random.randint(0,1)
        observation, reward, done, info = env.step(action)
        score+=reward
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()
print(score)
