import pdb

import numpy as np

from gradient import get_gradient, get_hessian


def gradient_descent(function, x, learning_rate, epsilon, max_iteration):
    history = []
    for itr in range(max_iteration):
        grad = get_gradient(function.evaluate, x)
        grad_norm = np.sum(grad ** 2) ** 0.5
        history.append([x[0], x[1], function.evaluate(x), grad_norm])

        x -= learning_rate * grad

        if np.all(np.abs(grad) < epsilon):
            break

    return history


def momentum(function, x, learning_rate, epsilon, max_iteration, beta=0.9):
    history = []
    v = np.zeros(2)
    for itr in range(1, max_iteration + 1):
        grad = get_gradient(function.evaluate, x)
        grad_norm = np.sum(grad ** 2) ** 0.5
        history.append([x[0], x[1], function.evaluate(x), grad_norm])

        # Momentum 更新式
        v = beta * v + (1 - beta) * grad
        x -= learning_rate * v

        if np.all(np.abs(grad) < epsilon):
            break

    return history

    
def adagrad(function, x, learning_rate, epsilon, max_iteration):
    history = []
    v = np.zeros(2)
    for itr in range(1, max_iteration + 1):
        grad = get_gradient(function.evaluate, x)
        grad_norm = np.sum(grad ** 2) ** 0.5
        history.append([x[0], x[1], function.evaluate(x), grad_norm])

        # AdaGrad 更新式
        v = v + np.square(grad)
        x -= learning_rate * (1 / np.sqrt(v)) * grad

        if np.all(np.abs(grad) < epsilon):
            break

    return history


def rmsprop(function, x, learning_rate, epsilon, max_iteration, beta=0.9):
    history = []
    v = np.zeros(2)
    for itr in range(1, max_iteration + 1):
        grad = get_gradient(function.evaluate, x)
        grad_norm = np.sum(grad ** 2) ** 0.5
        history.append([x[0], x[1], function.evaluate(x), grad_norm])

        # RMSProp 更新式
        v = beta * v + (1 - beta) * np.square(grad)
        x -= learning_rate * (1 / (np.sqrt(v) + epsilon)) * grad

        if np.all(np.abs(grad) < epsilon):
            break

    return history


def adam(function, x, learning_rate, epsilon, max_iteration, beta_1=0.9, beta_2=0.999):
    history = []
    m = np.zeros(2)
    v = np.zeros(2)
    for itr in range(1, max_iteration + 1):
        grad = get_gradient(function.evaluate, x)
        grad_norm = np.sum(grad ** 2) ** 0.5
        history.append([x[0], x[1], function.evaluate(x), grad_norm])

        # Adam 更新式
        learning_rate_next = learning_rate * (np.sqrt(1 - np.power(beta_2, itr)) / (1 - np.power(beta_1, itr)))

        m = beta_1 * m + (1 - beta_1) * grad
        v = beta_2 * v + (1 - beta_2) * np.square(grad)
        x -= learning_rate_next * m / (np.sqrt(v) + epsilon)

        if np.all(np.abs(grad) < epsilon):
            break
        
    return history


def conjugate_gradient(function, x, learning_rate, epsilon, max_iteration):
    history = []
    for itr in range(max_iteration):
        grad = get_gradient(function.evaluate, x)
        grad_norm = np.sum(grad ** 2) ** 0.5
        history.append([x[0], x[1], function.evaluate(x), grad_norm])

        if itr == 0:
            x -= learning_rate * grad
            pre_negative_grad = -grad
            pre_s = -grad
            continue

        negative_grad = -grad

        beta = (negative_grad @ negative_grad - negative_grad @ pre_negative_grad) / (pre_negative_grad @ pre_negative_grad)
        s = negative_grad + beta * pre_s
        x += learning_rate * s

        pre_negative_grad = negative_grad
        pre_s = s

        if np.all(np.abs(grad) < epsilon):
            break

    return history


def newton_method(function, x, learning_rate, epsilon, max_iteration):
    history = []
    for itr in range(max_iteration):
        grad = get_gradient(function.evaluate, x)
        hessian = get_hessian(function.evaluate, x)

        grad_norm = np.sum(grad ** 2) ** 0.5
        history.append([x[0], x[1], function.evaluate(x), grad_norm])

        x -= np.linalg.inv(hessian) @ grad

        if np.all(np.abs(grad) < epsilon):
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