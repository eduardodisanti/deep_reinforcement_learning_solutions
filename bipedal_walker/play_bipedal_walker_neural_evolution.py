import numpy as np
import pickle
import gym
from models.ANN import ANN2

with open("bipedal_walker_ne_200_128X24.h5", "rb") as f:
    M = pickle.load(f)

model = M['model']
rew = M['reward']
eps = M['episodes']

print("episodes", eps, "res avg", np.average(rew))

ENVIRONMENT = 'BipedalWalker-v2'
env = gym.make(ENVIRONMENT)

for t in range(20):
    done = False
    state = env.reset()
    score = 0
    while not done:
        # get the action
        action = model.sample_action(state)

        # perform the action
        #action = np.argmax(action)
        state, reward, done, _ = env.step(action)
        score+=reward
        env.render()
    print("Game", t, "reward", score)
