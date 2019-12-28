import gym

from lunar_lander.dqn_lunar_lander_agent import Agent

env = gym.make('LunarLander-v2')
env.reset()
score = 0

action_size = env.action_space.n
state_size = env.observation_space.shape[0]

def choose_action(state, agent, eps=0.):
    action = agent.act(state, eps=eps)

    return action

agent = Agent(state_size=state_size, action_size=action_size)

agent.load_model("lunar_lander.pt")

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
