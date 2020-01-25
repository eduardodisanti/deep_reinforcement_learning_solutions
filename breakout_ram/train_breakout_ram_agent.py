import os
from collections import deque

import gym
import torch
import numpy as np

import matplotlib.pyplot as plt

from env_wrappers.atari_wrappers import make_atari, Monitor, wrap_deepmindRAM
from models.dqn_agent import Agent
from auxs.aux_funcs import create_action_dict, choose_action


#env = gym.make('Breakout-ram-v0')

env = make_atari('Breakout-ram-v0', skip=1)
env = wrap_deepmindRAM(env, frame_stack=False, clip_rewards=True, episode_life=True)
env = Monitor(env)

env.reset()

action_size = env.action_space.n
state_size = env.observation_space.shape[0]

print("state size", state_size)

EPS_START = 1           # START EXPLORING A LOT
GAMMA = 0.1             # discount factor -

BUFFER_SIZE = int(1e2)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
TAU = 1e-2              # for soft update of target parameters
LR = 7e-4               # learning rate
UPDATE_EVERY = 5        # how often to update the network
eps_min = 0.0005        # NOT EXPLORE MUCH AFTER DECAY
eps_decay = 0.99995       # DECAY EXPLORE SLOWLY

agent = Agent(state_size=state_size, action_size=action_size, seed=0, gamma=GAMMA, buffer_size=BUFFER_SIZE,
              batch_size=BATCH_SIZE, tau=TAU, lr=LR, update_every=UPDATE_EVERY, fc1_neurons=1024, fc2_neurons=1024)

TARGET_AVG_SCORE = 10
NUM_OF_TARGET_EPISODES_FOR_AVG = 100

SHOW_TRAIN = 2000
trained = False
episodes = 0
la = create_action_dict(action_size)
lq = deque(maxlen=1000)
consecutives_solved = 0
times_solved = 0
avg = 0
mav = 0
avgs = deque(maxlen=1000)
mavgs = deque(maxlen=1000)

scores = deque(maxlen=1000)

eps = EPS_START

max_score = -10e10

episode_eps_min = None

while not trained:
    state = env.reset()  # reset the environment
    score = 0  # initialize the score
    lives_ant = env.episodic_env.lives
    while True:
        if episodes >= SHOW_TRAIN:
            env.render()
        if eps <= eps_min and episode_eps_min == None:
            episode_eps_min = episodes
        action = choose_action(state, agent, eps)  # select an action
        la[action] += 1
        next_state, reward, done, info = env.step(action)
        if lives_ant > env.episodic_env.lives:
            reward -= 1
            lives_ant = env.episodic_env.lives

        score+=reward
        if done:
            #print("Episode", episodes, info)
            break
        agent.step(state, action, reward, next_state, done)
        eps = max(eps_min, eps_decay * eps)
        state = next_state  # roll over the state to next time step

    lq.append(score)
    avg = np.average(lq)
    avgs.append(avg)
    episodes += 1
    #print("Episodes", episodes, "last score", score)

    if (len(avgs)%1000) == 0:
        #if episodes <= SHOW_TRAIN :
        plt.clf()
        plt.plot(avgs, c="r")
        plt.plot(lq, "x", c="b")
        plt.title("Episodes " + str(episodes))
#        if episode_eps_min:
#            plt.vlines(episode_eps_min, 0, max_score)
        plt.pause(0.1)
        print("act", la)
        print("episodes", episodes, "last score", score, "current eps", eps, "avg", avg)
        if avg > max_score:
            torch.save(agent.qnetwork_local.state_dict(), 'breakout_ram.pt')
            max_score = avg

    if (avg > TARGET_AVG_SCORE and episodes > NUM_OF_TARGET_EPISODES_FOR_AVG) or episodes > 100000000:
        trained = True
        torch.save(agent.qnetwork_local.state_dict(), 'breakout_ram.pt')
        print("Trained")
        print("episodes", episodes, "last score", score, "current eps", eps,  "avg", avg)

plt.savefig("breakout_ram_train_history.png")
print("Score: {}".format(score))
