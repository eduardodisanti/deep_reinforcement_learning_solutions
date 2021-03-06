import numpy as np
import gym
import os
import pickle
import matplotlib.pyplot as plt

from datetime import datetime
from multiprocessing.dummy import Pool

# thread pool for parallelization
from env_wrappers.atari_wrappers import make_atari, wrap_deepmindRAM, Monitor
from models.ANN import ANN

cpus = os.cpu_count()
print("CPUS", cpus)
pool = Pool(cpus)

### neural network

# hyperparameters
ENVIRONMENT = 'Breakout-ram-v0'

env = make_atari(ENVIRONMENT, skip=4)
env = wrap_deepmindRAM(env, frame_stack=False, clip_rewards=True, episode_life=True)
env = Monitor(env)

env.reset()

action_size = env.action_space.n

state_size = env.observation_space.shape[0]
D = state_size
M1 = 1024
K = env.action_space.n
action_max = env.action_space.n

def save_model_params(NN, rewards, generations):

    with open("breakout_ne_64x24.h5", "wb") as f:
        pickle.dump({'model':NN, 'reward':rewards, 'episodes':generations}, f)

def evolution_strategy(
        f,
        population_size,
        sigma,
        lr,
        initial_params,
        target_score=200,
        target_episodes=10):
    # assume initial params is a 1-D array
    num_params = len(initial_params)
    reward_per_iteration = []

    params = initial_params
    t = 0
    trained = False
    consecutively_solved = 0
    while not trained:
        t0 = datetime.now()
        N = np.random.randn(population_size, num_params)

        ### fast way
        R = pool.map(f, [params + sigma * N[j] for j in range(population_size)])
        R = np.array(R)

        m = R.mean()
        s = R.std()
        if s == 0:
            # we can't apply the following equation
            print("Skipping")
            continue

        A = (R - m) / s
        reward_per_iteration.append(m)
        params = params + lr / (population_size * sigma) * np.dot(N.T, A)

        # update the learning rate
        # lr *= 0.992354
        # sigma *= 0.99

        print("Episode:", t, "avg: %.4f" % m, "max:", R.max(), "time:", (datetime.now() - t0))
        model = get_NN()
        model.set_params(params)
        save_model_params(model, R.max(), t)

        if m >= target_score:
            consecutively_solved += 1
        else:
            consecutively_solved = 0

        if consecutively_solved >= target_episodes:
            trained = True
        t += 1

    return params, reward_per_iteration


def reward_function(params, display=False):
    model = ANN(D, M1, K, action_max)
    model.set_params(params)

    # play one episode and return the total reward
    episode_reward = 0
    episode_length = 0  # not sure if it will be used
    done = False
    env = make_atari(ENVIRONMENT, skip=4)
    env = wrap_deepmindRAM(env, frame_stack=False, clip_rewards=True, episode_life=True)
    env = Monitor(env)
    state = env.reset()

    lives = env.episodic_env.lives
    while not done:
        # get the action
        action = model.sample_action(state)

        # perform the action
        action = np.argmax(action)
        state, reward, done, _ = env.step(action)
        if env.episodic_env.lives < lives:
            lives = env.episodic_env.lives
            reward-=1

        # update total reward
        episode_reward += reward
        episode_length += 1

    return episode_reward

def get_NN():
    return(model)

if __name__ == '__main__':
    model = ANN(D, M1,K, action_max)

    model.init()
    params = model.get_params()
    best_params, rewards = evolution_strategy(
        f=reward_function,
        population_size=100,
        sigma=0.1,
        lr=0.03,
        initial_params=params,
        target_score = 200,
        target_episodes = 100
    )

    model.set_params(best_params)
    save_model_params(model, rewards, len(rewards))
