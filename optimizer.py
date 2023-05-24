import pdb

import numpy as np

from gradient import central_difference


def gradient_descent(function, initial_point, learning_rate, epsilon, max_iteration):
    x, y = initial_point
    history = []

    for itr in range(max_iteration):
        grad_x, grad_y = central_difference(function.evaluate, x, y)
        grad = (grad_x ** 2 + grad_y ** 2) ** 0.5
        history.append([x, y, function.evaluate(x, y), grad])

        x -= learning_rate * grad_x
        y -= learning_rate * grad_y

        if abs(grad_x) < epsilon and abs(grad_y) < epsilon:
            break

    return history


'''
def stochastic_gradient_descent(function, initial_point, learning_rate, epsilon, max_iteration):
    x, y = initial_point
    history = []

    for itr in range(max_iteration):
        x_i, y_i = function.generate_random_point()
        grad_x, grad_y = central_difference(function.evaluate, x_i, y_i)
        grad = (grad_x ** 2 + grad_y ** 2) ** 0.5
        history.append([x, y, function.evaluate(x, y), grad])

        x -= learning_rate * grad_x
        y -= learning_rate * grad_y

        if abs(grad_x) < epsilon and abs(grad_y) < epsilon:
            break

    return history
    '''


def momentum(function, initial_point, learning_rate, epsilon, max_iteration, beta=0.9):
    x, y = initial_point
    history = []

    v = np.zeros(2)
    for itr in range(1, max_iteration + 1):
        grad_x, grad_y = central_difference(function.evaluate, x, y)
        grad = (grad_x ** 2 + grad_y ** 2) ** 0.5
        history.append([x, y, function.evaluate(x, y), grad])

        x_np = np.array([x, y])
        grad_np = np.array([grad_x, grad_y])

        # Momentum 更新式
        v = beta * v + (1 - beta) * grad_np

        x_np -= learning_rate * v
        x, y = x_np

        if abs(grad_x) < epsilon and abs(grad_y) < epsilon:
            break

    return history

    
def adagrad(function, initial_point, learning_rate, epsilon, max_iteration):
    x, y = initial_point
    history = []

    v = np.zeros(2)
    for itr in range(1, max_iteration + 1):
        grad_x, grad_y = central_difference(function.evaluate, x, y)
        grad = (grad_x ** 2 + grad_y ** 2) ** 0.5
        history.append([x, y, function.evaluate(x, y), grad])

        x_np = np.array([x, y])
        grad_np = np.array([grad_x, grad_y])

        # AdaGrad 更新式
        v = v + np.square(grad_np)
        
        x_np -= learning_rate * (1 / np.sqrt(v)) * grad_np
        x, y = x_np

        if abs(grad_x) < epsilon and abs(grad_y) < epsilon:
            break

    return history


def rmsprop(function, initial_point, learning_rate, epsilon, max_iteration, beta=0.9):
    x, y = initial_point
    history = []

    v = np.zeros(2)
    for itr in range(1, max_iteration + 1):
        grad_x, grad_y = central_difference(function.evaluate, x, y)
        grad = (grad_x ** 2 + grad_y ** 2) ** 0.5
        history.append([x, y, function.evaluate(x, y), grad])

        x_np = np.array([x, y])
        grad_np = np.array([grad_x, grad_y])

        # RMSProp 更新式
        v = beta * v + (1 - beta) * np.square(grad_np)

        x_np -= learning_rate * (1 / (np.sqrt(v) + epsilon)) * grad_np
        x, y = x_np

        if abs(grad_x) < epsilon and abs(grad_y) < epsilon:
            break

    return history


def adam(function, initial_point, learning_rate, epsilon, max_iteration, beta_1=0.9, beta_2=0.999):
    x, y = initial_point
    history = []

    m = np.zeros(2)
    v = np.zeros(2)
    for itr in range(1, max_iteration + 1):
        grad_x, grad_y = central_difference(function.evaluate, x, y)
        grad = (grad_x ** 2 + grad_y ** 2) ** 0.5
        history.append([x, y, function.evaluate(x, y), grad])

        x_np = np.array([x, y])
        grad_np = np.array([grad_x, grad_y])

        # Adam 更新式
        learning_rate_next = learning_rate * (np.sqrt(1 - np.power(beta_2, itr)) / (1 - np.power(beta_1, itr)))

        m = beta_1 * m + (1 - beta_1) * grad_np
        v = beta_2 * v + (1 - beta_2) * np.square(grad_np)

        x_np -= learning_rate_next * m / (np.sqrt(v) + epsilon)
        x, y = x_np

        if abs(grad_x) < epsilon and abs(grad_y) < epsilon:
            break
        
    return history


class Optimizer:
    def __init__(self, optimizer, learning_rate=0.01, epsilon=1e-5, max_iteration=30000):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iteration = max_iteration

    def optimize(self, function):
        initial_point = function.generate_initial_point()
        result = self.optimizer(function, initial_point, self.learning_rate, self.epsilon, self.max_iteration)

        return result