from collections import deque

import gym
import torch
import numpy as np

import matplotlib.pyplot as plt

from models.ddpg_agent import Agent

env = gym.make('BipedalWalker-v2')
env.reset()

action_size = env.action_space.shape[0]
state_size = env.observation_space.shape[0]

N_EPISODES      = 500
MAX_T           = 2000
TARGET_SCORE    = 300.0
TARGET_EPISODES = 1600

SHOW_TRAIN = 300

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

    if  np.mean(scoresDQ) > best_score:
        torch.save(agent.actor_local.state_dict(), actor_path)
        torch.save(agent.critic_local.state_dict(), critic_path)

    if np.average(scoresDQ) >= TARGET_SCORE:
        print("Environment SOLVED in " + str(episode_ist) + " episodes")
        print("Average = " + str(np.mean(scoresDQ)) + " over last " + str(TARGET_EPISODES) + " episodes")

        torch.save(agent.actor_local.state_dict(), actor_path)
        torch.save(agent.critic_local.state_dict(), critic_path)
        plt.savefig("bipedal_walker_train_history.png")
        break
