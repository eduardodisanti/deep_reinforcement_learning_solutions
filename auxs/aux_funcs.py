import numpy as np

def create_action_dict(n):

    res = {}
    for i in range(n):
        res[i] = 0

    return res

def onehot_encode(i, n):

    v = np.zeros(n)
    v[i] = 1

    #return(np.array([i]))
    return(v)

def choose_action(state, agent, eps=0.):
    action = agent.act(state, eps=eps)

    return action
