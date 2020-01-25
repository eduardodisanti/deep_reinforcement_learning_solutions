import time
import numpy as np
import gym
env = gym.make('Breakout-ram-v0')
env = env.unwrapped
env.reset()
score = 0

action_size = env.action_space.n
state_size = env.observation_space.shape[0]

print("Action size", action_size, "state size", state_size)

scores = []
for i_episode in range(100):
    observation = env.reset()
    score = 0
    while True:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        score+=reward
        scores.append(score)
        if done:
            print("Episode finished {} score".format(score))
            print("Average score {}".format(np.average(scores)))
            break
        #time.sleep(1 / (20 * (i_episode + 1)))
env.close()
print(score)
