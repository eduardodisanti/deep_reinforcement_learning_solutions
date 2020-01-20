import os
from collections import deque

import gym
import torch
import numpy as np

import matplotlib.pyplot as plt

from models.ddpgagent import DDPGAgent

env = gym.make('BipedalWalker-v2')
env.reset()

action_size = env.action_space.shape[0]
state_size = env.observation_space.shape[0]

N_EPISODES      = 1000
MAX_T           = 1000
TARGET_SCORE    = 300.0
TARGET_EPISODES = 100

SHOW_TRAIN = 20

agent = DDPGAgent(state_size=state_size, action_size=action_size, random_seed=1,
                         fc1_actor_units=300,
                         fc2_actor_units=300,
                         fc1_critic_units=300,
                         fc2_critic_units=300)

actor_path = "bipedal_actor.pt"
critic_path= "bipedal_critic.pt"

episode_ist = 1
best_score = -1e10

if os.path.isfile(actor_path) and os.path.isfile(critic_path):
    agent.load_model(actor_path, critic_path)
    episode_ist = SHOW_TRAIN
    best_score = -110.0

train_mode = True
scoresDQ = deque(maxlen=TARGET_EPISODES)  # mean scores of n most recent episodes (n=target_episodes)
avgs = []

train_show_window = 10
t = 0
state = env.reset()  # reset environment
states = np.array([state])  # TRANSFORM IN A LIST BECAUSE THE DDPG AGENT WAS IMPLEMENTED FOR MULTI AGENT TRAINING #
agents_score = np.zeros(1)  # initialize score for the agent
agent.reset()

while True:
    #for t in range(MAX_T):
    rew_list = []
    #while True:
    if True:
        if episode_ist > SHOW_TRAIN and episode_ist <= SHOW_TRAIN + train_show_window:
            env.render(mode="human")
        elif episode_ist > SHOW_TRAIN and episode_ist > SHOW_TRAIN + train_show_window:
             SHOW_TRAIN = SHOW_TRAIN + 2 * train_show_window
             env.render(mode="machine")
        t+=1
        actions = agent.act(states, add_noise=True)  # select an action for agents
        next_state, reward, done, info = env.step(actions[0])  # send actions to environment

        if reward == 0:   # IN CASE THE AGENT DECIDES TO DO NOTHING
            reward = -0.1
        dones = np.array([done])  # see if episode has finished
        if t >= MAX_T:
            #if rew_list[:-1] < 0:
            #    reward-=100         # IN CASE THE AGENT DECIDES JUST TO STOP RECEIVES A NEGATIVE REWARD
            dones[0] = True

        rew_list.append(reward)
        next_states = np.array([next_state])
        rewards = np.array([reward])  # get the rewards

        # save experience to replay buffer, perform learning step at defined interval
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            agent.step(state, action, reward, next_state, done, t)
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

    if  np.mean(scoresDQ) > best_score or episode_ist %1000 == 0:
        torch.save(agent.actor_local.state_dict(), actor_path)
        torch.save(agent.critic_local.state_dict(), critic_path)

    if np.average(scoresDQ) >= TARGET_SCORE:
        print("Environment SOLVED in " + str(episode_ist) + " episodes")
        print("Average = " + str(np.mean(scoresDQ)) + " over last " + str(TARGET_EPISODES) + " episodes")

        torch.save(agent.actor_local.state_dict(), actor_path)
        torch.save(agent.critic_local.state_dict(), critic_path)
        plt.savefig("bipedal_walker_train_history.png")
        break
