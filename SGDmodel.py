# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    SGDmodel.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: yohan <yohan@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/11 18:34:17 by yohan             #+#    #+#              #
#    Updated: 2025/02/12 13:41:19 by yohan            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

class AdalineSGD(object):
    """Adaptive linear neuron classifier (binary classifier) using SGD for performance advantage"""
    
    # learning_rate : bool -> learning rate
    # n_iter : int         -> number of epochs of learning
    # shuffle : bool       -> (default : true) Used to shuffle training data to prevent cycles
    # random_state : int   -> ranNumGen seed for weight initialization
    
    # w_initialized : bool -> flag to check if weights have been initialized
    # w_ : array           -> 1D array of unknown weights
    # cost_ : list         -> array of costs in accordance to weight iteration (eg. weight[j] has cost[j] cost)
    
    # X : vector           -> Vector of inputs (features)
    # y : vector           -> Target values (training set)
    
    def __init__(self, learning_rate=0.01, n_iter=10, shuffle=True, random_state=None):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.w_initialized = False
        self.w_ = None
        self.cost_ = []
        
    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal (loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] +=  self.learning_rate * xi.dot(error)
        self.w_[0] += self.learning_rate * error
        cost = 0.5 * error ** 2
        return cost

    def activation(self, X, alpha=0.01):
        # return np.where(X > 0, X, alpha * X) # leaky ReLu activation
        return X
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0] # sum of all weights + bias

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def partial_fit(self, X, y): # used for online learning scenarios where we receive one data after the other
        """training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self
    
    def fit(self, X, y):    #used to train machine with initial dataset
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y) # getting mean cost
            self.cost_.append(avg_cost)
        return self