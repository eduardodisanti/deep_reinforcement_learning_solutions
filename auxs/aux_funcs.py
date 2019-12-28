import numpy as np

def create_action_dict(n):

    res = {}
    for i in range(n):
        res[i] = 0

    return res

def onehot_encode(i, n):

    v = np.zeros(n)
    v[i] = 1

    return(v)