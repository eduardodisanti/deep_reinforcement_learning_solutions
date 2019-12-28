import gym

from auxs.aux_funcs import onehot_encode
from taxi_v3.dqn_taxiv3_agent import Agent

env = gym.make('Taxi-v2')
env.reset()
score = 0

action_size = env.action_space.n
state_size = env.observation_space.n

def choose_action(state, agent, eps=0.):
    action = agent.act(state, eps=eps)

    return action

agent = Agent(state_size=state_size, action_size=action_size)

agent.load_model("taxi_v3.pt")

total_score = 0
for i_episode in range(20):
    observation = env.reset()
    observation = onehot_encode(observation, state_size)
    score = 0
    while True:
        env.render()
        action = choose_action(observation, agent, 0)
        observation, reward, done, info = env.step(action)
        observation = onehot_encode(observation, state_size)
        score+=reward
        if done:
            print("Episode finished with score {} ".format(score))
            total_score += score
            break

print("Episode finished with average score {} ".format(total_score / i_episode))
env.close()
print(score)
