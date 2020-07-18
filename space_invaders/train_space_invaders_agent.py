import gym
import torch
import numpy as np

import matplotlib.pyplot as plt

from auxs.aux_funcs import choose_action, create_action_dict
from models.dqn_agent import Agent

env = gym.make('SpaceInvaders-ram-v0')
env.reset()

env.reset()

action_size = env.action_space.n
action_size = 3
state_size = env.observation_space.shape[0]

EPS_START = 1  # START EXPLORING A LOT
GAMMA = 0.99999  # discount factor -

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
eps_min = 0.005  # EVEN EXPLORE AFTER MANY EPISODES
eps_decay = 0.99995  # DECAY EXPLORE SLOWLY

agent = Agent(state_size=state_size, action_size=action_size, seed=0, gamma=GAMMA, buffer_size=BUFFER_SIZE,
              batch_size=BATCH_SIZE, tau=TAU, lr=LR, update_every=UPDATE_EVERY, fc1_neurons=128, fc2_neurons=128)

TARGET_AVG_SCORE = 800
NUM_OF_TARGET_EPISODES_FOR_AVG = 100

trained = False
episodes = 0
la = create_action_dict(action_size)
lq = []
consecutives_solved = 0
times_solved = 0
avg = 0
mav = 0
avgs = []
mavgs = []

scores = []

eps = EPS_START

max_score = -10e10

episode_eps_min = None

while not trained:
    state = env.reset()  # reset the environment
    score = 0  # initialize the score
    while True:
        if eps <= eps_min and episode_eps_min == None:
            episode_eps_min = episodes
        action = choose_action(state, agent, eps)  # select an action
        la[action] += 1
        next_state, reward, done, info = env.step(action)

        score+=reward
        if done:
            break
        agent.step(state, action, reward, next_state, done)
        eps = max(eps_min, eps_decay * eps)
        state = next_state  # roll over the state to next time step

        if episodes > 1000:
            env.render()

    episodes += 1
    lq.append(score)
    avg = np.average(lq[-NUM_OF_TARGET_EPISODES_FOR_AVG:])
    avgs.append(avg)

    if (len(avgs)%10) == 0:
        plt.plot(avgs, c="r")
        plt.plot(lq, ".", c="b")
        if episode_eps_min:
            plt.vlines(episode_eps_min, 0, max_score)
        plt.pause(0.1)
        print("act", la)
        print("episodes", episodes, "last score", score, "current eps", eps, "avg", avg)
        if avg > max_score:
            torch.save(agent.qnetwork_local.state_dict(), 'space_invaders.pt')
            max_score = avg
    if (avg > TARGET_AVG_SCORE and episodes > NUM_OF_TARGET_EPISODES_FOR_AVG) or episodes > 100000:
        trained = True
        torch.save(agent.qnetwork_local.state_dict(), 'space_invaders.pt')
        print("Trained")
        print("episodes", episodes, "last score", score, "current eps", eps,  "avg", avg)

    if episodes > 3000:
        break
plt.savefig("space_invaders.png")
print("Score: {}".format(score))