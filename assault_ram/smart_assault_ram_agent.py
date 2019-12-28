import time

import gym
import random
import torch

from assault_ram.dqn_agent import Agent

env = gym.make('Assault-ram-v0')
env.reset()
score = 0


def choose_action(state, agent, eps=0.):
    action = agent.act(state, eps=eps)

    return action

action_size = env.action_space.n
state_size = env.observation_space.shape[0]

agent = Agent(state_size=state_size, action_size=action_size)

agent.load_model("assault_ram.pt")

total_score = 0
for i_episode in range(20):
    observation = env.reset()
    score = 0
    while True:
        env.render(mode='human')
        action = choose_action(observation, agent, 0)
        observation, reward, done, info = env.step(action)
        score+=reward
        if done:
            print("Episode finished with score {} ".format(score))
            total_score += score
            break

        time.sleep(1/(10 * i_episode))

print("Episode finished with average score {} ".format(total_score / i_episode))
env.close()
print(score)
