import gym
import numpy as np

from auxs.aux_funcs import choose_action
from models.dqn_agent import Agent

env = gym.make('Acrobot-v1')
env.reset()
score = 0

action_size = env.action_space.n
state_size = env.observation_space.shape[0]

agent = Agent(state_size=state_size, action_size=action_size, fc1_neurons=128, fc2_neurons=128)


agent.load_model("acrobot.pt")

total_score = 0
for i_episode in range(20):
    observation = env.reset()
    score = 0
    while True:
        env.render()
        observation = np.array([observation])
        action = choose_action(observation, agent)
        observation, reward, done, info = env.step(action)
        score+=reward
        if done:
            print("Episode finished with score {} ".format(score))
            total_score += score
            break

print("Episode finished with average score {} ".format(total_score / i_episode))
env.close()
print(score)
