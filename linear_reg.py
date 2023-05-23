import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, data, target, learning_rate=0.01, iterations=1000):
        self.data = data
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.target = target

    def _rescale(self):
        rescaled_data = self.data.copy()
        rescaled_data = (rescaled_data - rescaled_data.mean()) / \
            rescaled_data.std()
        rescaled_data.insert(0, 'ones', 1)
        df = rescaled_data
        cols = df.shape[1]
        X = df.iloc[:, 0:cols-1]
        y = df.iloc[:, cols-1:cols]

        theta = np.matrix(np.array([0]*X.shape[1]))

        X = np.matrix(X.values)
        y = np.matrix(y.values)

        self.X = X
        self.y = y
        self.theta = theta
        return X, y, theta

    def _compute_cost(self, X, y, theta):
        z = np.power(((X*theta.T)-y), 2)
        return np.sum(z)/(2*len(X))

    def _gradient_descent(self, X, y, theta, learning_rate, iterations):
        temp = np.matrix(np.zeros(theta.shape))
        parameters = int(theta.ravel().shape[1])
        cost = np.zeros(iterations)

        for i in range(iterations):
            error = (X*theta.T)-y

            for j in range(parameters):
                term = np.multiply(error, X[:, j])
                temp[0, j] = theta[0, j] - \
                    ((learning_rate/len(X))*np.sum(term))
            theta = temp
            cost[i] = self._compute_cost(X, y, theta)

        return theta, cost

    def fit(self):
        X, y, theta = self._rescale()
        g, cost = self._gradient_descent(
            X, y, theta, self.learning_rate, self.iterations)

        print("Model trained successfully !!!!")
        self.g = g
        self.cost = cost
        return

    def predict(self, X):
        X = np.matrix(X)
        y_pred = X*self.g.T
        return y_pred
