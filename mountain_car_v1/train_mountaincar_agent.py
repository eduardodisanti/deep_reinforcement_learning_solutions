import gym
import torch
import numpy as np

import matplotlib.pyplot as plt

from auxs.aux_funcs import choose_action
from models.dqn_agent import Agent

env = gym.make('MountainCar-v0')
env.reset()

action_size = env.action_space.n
state_size = env.observation_space.shape[0]

EPS_START = 1  # START EXPLORING A LOT
GAMMA = 0.999  # discount factor -

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

agent = Agent(state_size=state_size, action_size=action_size, seed=0, gamma=GAMMA, buffer_size=BUFFER_SIZE,
              batch_size=BATCH_SIZE, tau=TAU, lr=LR, update_every=UPDATE_EVERY, fc1_neurons=200, fc2_neurons=200)

TARGET_AVG_SCORE = -50
NUM_OF_TARGET_EPISODES_FOR_AVG = 100

eps_min = 0.001  # EVEN EXPLORE AFTER MANY EPISODES
eps_decay = 0.99995  # DECAY EXPLORE SLOWLY


trained = False
episodes = 0
la = {0: 0, 1: 0, 2:0}
lq = []
consecutives_solved = 0
times_solved = 0
avg = 0
mav = 0
avgs = []
mavgs = []

scores = []

eps = EPS_START

while not trained:
    state = env.reset()  # reset the environment
    score = 0  # initialize the score
    #for trys in range(0,20000):
    while True:
        #env.render()
        action = choose_action(state, agent, eps)  # select an action
        la[action] += 1
        next_state, reward, done, info = env.step(action)

        if next_state[0]> -0.2:  #### ADJUST THE REWARD FOR FASTEST LEARNING ####
            reward = 1
        score += reward # update the score
        if done:  # exit loop if episode finished
            break

        agent.step(state, action, reward, next_state, done)
        eps = max(eps_min, eps_decay * eps)
        state = next_state  # roll over the state to next time step

    episodes += 1
    lq.append(score)

    avg = np.average(lq[-NUM_OF_TARGET_EPISODES_FOR_AVG:])
    avgs.append(avg)

    if (len(avgs)%10) == 0:
        plt.plot(avgs, c="b")
        plt.pause(0.1)
        print("act", la)
        print("episodes", episodes, "last score", score, "current eps", eps, "solved", times_solved, "avg", avg)
        torch.save(agent.qnetwork_local.state_dict(), 'mountain_car_v0_chk.pt')

    if avg > TARGET_AVG_SCORE:
        times_solved += 1
    else:
        consecutives_solved = 0
    if avg > TARGET_AVG_SCORE:
        trained = True
        torch.save(agent.qnetwork_local.state_dict(), 'mountain_car_v0.pt')
        print("Trained")
        print("episodes", episodes, "last score", score, "current eps", eps, "solved", times_solved, "avg", avg)

print("Score: {}".format(score))
