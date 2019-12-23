import gym
import random
import torch

from cartpole.dqn_agent import Agent

env = gym.make('CartPole-v0')
env.reset()
score = 0


def choose_action(state, agent, eps=0.):
    action = agent.act(state, eps=eps)

    return action

agent = Agent()

agent.load_model("cart_pole.pt")

total_score = 0
for i_episode in range(20):
    observation = env.reset()
    score = 0
    while True:
        env.render()
        action = choose_action(observation, agent, 0)
        observation, reward, done, info = env.step(action)
        score+=reward
        if done:
            print("Episode finished with score {} ".format(score))
            total_score += score
            break

print("Episode finished with average score {} ".format(total_score / i_episode))
env.close()
print(score)
