from collections import deque
import gym
import torch
import numpy as np

import matplotlib.pyplot as plt

from models.ddpg_agent import Agent
from auxs.aux_funcs import create_action_dict, choose_action

env = gym.make('Breakout-ram-v0')
env.reset()

action_size = env.action_space.n
state_size = env.observation_space.shape[0]

print("state size", state_size)
EPS_START = 1  # START EXPLORING A LOT
GAMMA = 0.9999  # discount factor -

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 8  # how often to update the network
eps_min = 0.005  # EVEN EXPLORE AFTER MANY EPISODES
eps_decay = 0.999995  # DECAY EXPLORE SLOWLY

SHOW_TRAIN = 300
N_EPISODES      = 100
MAX_T           = 100000
TARGET_SCORE    = 300.0
TARGET_EPISODES = 1600


agent = Agent(state_size=state_size, action_size=action_size, random_seed=1)

actor_path = "bipedal_actor.pt"
critic_path= "bipedal_critic.pt"

train_mode = True
best_score = -1e10
scoresDQ = deque(maxlen=TARGET_EPISODES)  # mean scores of n most recent episodes (n=target_episodes)
avgs = []

episode_ist = 1

t = 0
state = env.reset()  # reset environment
states = np.array([state])  # TRANSFORM IN A LIST BECAUSE THE DDPG AGENT WAS IMPLEMENTED FOR MULTI AGENT TRAINING #
agents_score = np.zeros(1)  # initialize score for the agent
agent.reset()
lq = []
while True:
    #for t in range(MAX_T):
    rew_list = []
    #while True:
    if True:
        if episode_ist > SHOW_TRAIN:
            env.render()
        t+=1
        actions = agent.act(states, add_noise=True)  # select an action for agents
        next_state, reward, done, info = env.step(actions[0])  # send actions to environment

        dones = np.array([done])  # see if episode has finished
        if t >= MAX_T:
            reward-=100
            dones.append(True)

        rew_list.append(reward)
        next_states = np.array([next_state])
        rewards = np.array([reward])  # get the rewards

        # save experience to replay buffer, perform learning step at defined interval
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            agent.step(state, action, reward, next_state, done)
        states = next_states
        agents_score += rewards

        if np.any(dones) :  # breaks when any agent ends its episode       or t > MAX_T
            print("Done in", t, "score is", agents_score, rew_list[-1])
            t = 0
            scoresDQ.append(np.mean(agents_score))
            avgs.append(np.average(scoresDQ))

            state = env.reset()  # reset environment
            states = np.array([state])  # TRANSFORM IN A LIST BECAUSE THE DDPG AGENT WAS IMPLEMENTED FOR MULTI AGENT TRAINING #
            agents_score = np.zeros(1)  # initialize score for the agent
            agent.reset()
            if episode_ist < SHOW_TRAIN:
                plt.plot(avgs, ".", c="b")
                plt.pause(0.1)

            episode_ist += 1
            print("Episode", episode_ist, "average on deque", np.average(scoresDQ), "epsilon", agent.epsilon)
            print(len(scoresDQ), np.mean(agents_score), TARGET_SCORE, TARGET_EPISODES)

    lq.append(agents_score)
    if (len(avgs)%1) == 0:
        if episode_ist <= SHOW_TRAIN:
            plt.plot(avgs, c="r")
            plt.plot(lq, ".", c="b")
            plt.pause(0.1)
        print("episodes", episode_ist, "last score", agents_score, "current eps", agent.epsilon, "avg", np.average(scoresDQ))
        if np.average(scoresDQ) > max_score:
            torch.save(agent.qnetwork_local.state_dict(), 'breakout_ram.pt')
            max_score = np.average(scoresDQ)

    if (np.average(scoresDQ) > TARGET_SCORE and episode_ist > TARGET_EPISODES) or episode_ist > 100000:
        trained = True
        torch.save(agent.qnetwork_local.state_dict(), 'breakout_ram.pt')
        print("Trained")
        print("episodes", episode_ist, "last score", agents_score, "current eps", agent.epsilon,  "avg", np.average(scoresDQ))

plt.savefig("breakout_ram_train_history.png")
print("Score: {}".format(score))
