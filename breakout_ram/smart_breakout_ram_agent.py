import time
import numpy as np
import gym

from auxs.aux_funcs import choose_action
from models.dqn_agent import Agent

env = gym.make('Breakout-ram-v0')
env.reset()
score = 0


action_size = env.action_space.n
state_size = env.observation_space.shape[0]

agent = Agent(state_size=state_size, action_size=action_size, fc2_neurons=1024, fc1_neurons=1024)

agent.load_model("breakout_ram.pt")

total_score = 0
scores = []
for i_episode in range(20):
    observation = env.reset()
    score = 0
    while True:
        env.render(mode='human')
        action = choose_action(observation, agent, 0)
        observation, reward, done, info = env.step(action)
        score+=reward
        scores.append(score)
        if done:
            print("Episode finished {} score".format(score))
            print("Average score {}".format(np.average(scores)))
            break
        time.sleep(1 / (20 * (i_episode + 1)))

print("Episode finished with average score {} ".format(total_score / i_episode))
env.close()
print(score)
