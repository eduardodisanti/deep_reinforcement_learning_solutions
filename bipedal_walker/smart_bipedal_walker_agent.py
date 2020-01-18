import gym
import numpy as np

from models.a3c_agent import Agent

env = gym.make('BipedalWalker-v2')
env.reset()
score = 0

action_size = env.action_space.shape[0]
state_size = env.observation_space.shape[0]

def choose_action(state, agent):
    action = agent.act(state, add_noise=False)

    return action

agent = Agent(state_size=state_size, action_size=action_size, random_seed=1)


agent.load_model("bipedal_actor.pt", "bipedal_critic.pt")

total_score = 0
for i_episode in range(20):
    observation = env.reset()
    score = 0
    while True:
        env.render()
        observation = np.array([observation])
        action = choose_action(observation, agent)
        observation, reward, done, info = env.step(action[0])
        score+=reward
        if done:
            print("Episode finished with score {} ".format(score))
            total_score += score
            break

print("Episode finished with average score {} ".format(total_score / i_episode))
env.close()
print(score)
