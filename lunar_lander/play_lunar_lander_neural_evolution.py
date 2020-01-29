import numpy as np
import pickle
import gym
from models.ANN import ANN

with open("lunar_lander_ne_300x300.h5", "rb") as f:
    M = pickle.load(f)

model = M['model']

ENVIRONMENT = 'LunarLander-v2'
env = gym.make(ENVIRONMENT)

for t in range(20):
    done = False
    state = env.reset()
    score = 0
    while not done:
        # get the action
        action = model.sample_action(state)

        # perform the action
        action = np.argmax(action)
        state, reward, done, _ = env.step(action)
        score+=reward
        env.render()
    print("Game", t, "reward", score)
