import numpy as np
import gym
import os
import pickle
import matplotlib.pyplot as plt

from datetime import datetime
from multiprocessing.dummy import Pool

# thread pool for parallelization
from models.ANN import ANN2

cpus = os.cpu_count()
print("CPUS", cpus)
pool = Pool(cpus)

ENVIRONMENT = 'BipedalWalker-v2'
env = gym.make(ENVIRONMENT)

MODEL_NAME = "bipedal_walker_ne_200_64X16.h5"
generations = 300

if os.path.isfile(MODEL_NAME) and False:
    with open(MODEL_NAME, "rb") as f:
        M = pickle.load(f)
        model = M['model']
        rew = [M['reward']]
        initial_generations = M['episodes']
        generations = generations
else:
    env.reset()
    D = env.observation_space.shape[0]
    action_max = env.action_space.high

    M1 = 64
    M2 = 16
    K = env.action_space.shape[0]
    model = ANN2(D, M1, M2, K, action_max)
    model.init()

    rew = []

    params = model.get_params()


def save_model_params(NN, rewards, generations):

    with open("bipedal_walker_ne_200_64X16.h5", "wb") as f:
        pickle.dump({'model':NN, 'reward':rewards, 'episodes':generations}, f)

def evolution_strategy(
        f,
        population_size,
        sigma,
        lr,
        initial_params,
        generations,
        target_score=200,
        target_episodes=10,
        initial_timesteps=0,
        initial_rewards=[]):
    # assume initial params is a 1-D array
    num_params = len(initial_params)
    reward_per_iteration = initial_rewards.copy()

    params = initial_params
    t = initial_timesteps
    trained = False
    consecutively_solved = 0
    while not trained:
        t0 = datetime.now()
        N = np.random.randn(population_size, num_params)

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
    model = ANN2(D, M1, M2, K, action_max)
    model.set_params(params)

    env = gym.make(ENVIRONMENT)
#    params = model.get_params()

    # play one episode and return the total reward
    episode_reward = 0
    done = False
    state = env.reset()
    while not done:
        # get the action
        action = model.sample_action(state)

        state, reward, done, _ = env.step(action)

        # update total reward
        episode_reward += reward

    return episode_reward

def get_NN():
    return(model)

if __name__ == '__main__':
    best_params, rewards = evolution_strategy(
        f=reward_function,
        population_size=30,
        sigma=0.1,
        lr=0.03,
        initial_params=params,
        generations=generations,
        target_score = 200,
        target_episodes = 10,
        initial_rewards=rew
    )

    model.set_params(best_params)
    save_model_params(model, rewards, generations)
