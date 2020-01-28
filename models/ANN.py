import numpy as np

def relu(x):
    return x * (x > 0)

class ANN:
    def __init__(self, D, M, K, action_max, f=relu):
        self.D = D
        self.M = M
        self.K = K
        self.action_max = action_max
        self.f = f

    def init(self):
        D, M, K = self.D, self.M, self.K
        self.W1 = np.random.randn(D, M) / np.sqrt(D)
        # self.W1 = np.zeros((D, M))
        self.b1 = np.zeros(M)
        self.W2 = np.random.randn(M, K) / np.sqrt(M)
        # self.W2 = np.zeros((M, K))
        self.b2 = np.zeros(K)

    def forward(self, X):
        Z = self.f(X.dot(self.W1) + self.b1)
        return np.tanh(Z.dot(self.W2) + self.b2) * self.action_max

    def sample_action(self, x):
        # assume input is a single state of size (D,)
        # first make it (N, D) to fit ML conventions
        X = np.atleast_2d(x)
        Y = self.forward(X)
        return Y[0]  # the first row

    def get_params(self):
        # return a flat array of parameters
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])

    def get_params_dict(self):
        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
        }

    def set_params(self, params):
        # params is a flat list
        # unflatten into individual weights
        D, M, K = self.D, self.M, self.K
        self.W1 = params[:D * M].reshape(D, M)
        self.b1 = params[D * M:D * M + M]
        self.W2 = params[D * M + M:D * M + M + M * K].reshape(M, K)
        self.b2 = params[-K:]

