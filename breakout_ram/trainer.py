from env_wrappers.atari_wrappers import make_atari, Monitor, wrap_deepmindRAM
from models.dqn_agent import Agent
from auxs.aux_funcs import create_action_dict, choose_action


EPS_START = 1           # START EXPLORING A LOT
GAMMA = 0.1             # discount factor -

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
TAU = 1e-4              # for soft update of target parameters
LR = 7e-4               # learning rate
UPDATE_EVERY = 5        # how often to update the network
eps_min = 0.0005        # NOT EXPLORE MUCH AFTER DECAY
eps_decay = 0.99995       # DECAY EXPLORE SLOWLY


class RAM_Trainer:

    def __init__(self, gym_env_name='Breakout-ram-v0', skip=4):

        self.env = make_atari(gym_env_name, skip=skip)
        self.env = wrap_deepmindRAM(self.env, frame_stack=False, clip_rewards=True, episode_life=True)
        self.env = Monitor(self.env)

        self.env.reset()

        self.action_size = self.env.action_space.n
        self.state_size  = self.env.observation_space.shape[0]

        self.agent = Agent(state_size=self.state_size, action_size=self.action_size, seed=0, gamma=GAMMA, buffer_size=BUFFER_SIZE,
                      batch_size=BATCH_SIZE, tau=TAU, lr=LR, update_every=UPDATE_EVERY, fc1_neurons=512,
                      fc2_neurons=512)

        self.eps = EPS_START

        self.state = self.reset()
        self.episodes = 0
        self.done = False


    def train_episode(self):
        score = 0
        self.done = False
        self.rewards = []
        self.env.reset()
        while True:
            action = choose_action(self.state, self.agent, self.eps)  # select an action

            next_state, reward, self.done, info = self.env.step(action)

            score+=reward
            self.rewards.append(reward)
            if self.done:
               break

            self.agent.step(self.state, action, reward, next_state, self.done)
            self.eps = max(eps_min, eps_decay * self.eps)
            self.state = next_state  # roll over the state to next time step

        self.episodes += 1

    def reset(self):
        self.state = self.env.reset()
        self.rewards = []
        self.done = False

        return self.state
