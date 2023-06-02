import pdb

import numpy as np

from gradient import get_gradient, get_hessian
from vibration import vibration_analysis


def gradient_descent(function, x, learning_rate, epsilon, max_iteration, eigen_check):
    history = []
    for itr in range(max_iteration):
        grad = get_gradient(function.evaluate, x)
        grad_norm = np.sum(grad ** 2) ** 0.5
        history.append([x[0], x[1], function.evaluate(x), grad_norm])

        x -= learning_rate * grad

        if np.all(np.abs(grad) < epsilon):
            break

    eigen_val = vibration_analysis(function.evaluate, x, eigen_check)

    return history, grad, eigen_val


def momentum(function, x, learning_rate, epsilon, max_iteration, eigen_check, beta=0.9):
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

    eigen_val = vibration_analysis(function.evaluate, x, eigen_check)

    return history, grad, eigen_val

    
def adagrad(function, x, learning_rate, epsilon, max_iteration, eigen_check):
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

    eigen_val = vibration_analysis(function.evaluate, x, eigen_check)

    return history, grad, eigen_val


def rmsprop(function, x, learning_rate, epsilon, max_iteration, eigen_check, beta=0.9):
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

    eigen_val = vibration_analysis(function.evaluate, x, eigen_check)

    return history, grad, eigen_val


def adam(function, x, learning_rate, epsilon, max_iteration, eigen_check, beta_1=0.9, beta_2=0.999):
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

    eigen_val = vibration_analysis(function.evaluate, x, eigen_check)
       
    return history, grad, eigen_val


def adabelief(function, x, learning_rate, epsilon, max_iteration, eigen_check, beta_1=0.9, beta_2=0.999):
    history = []
    m = np.zeros(2)
    v = np.zeros(2)
    for itr in range(1, max_iteration + 1):
        grad = get_gradient(function.evaluate, x)
        grad_norm = np.sum(grad ** 2) ** 0.5
        history.append([x[0], x[1], function.evaluate(x), grad_norm])

        # AdaBelief 更新式
        learning_rate_next = learning_rate * (np.sqrt(1 - np.power(beta_2, itr)) / (1 - np.power(beta_1, itr)))

        m = beta_1 * m + (1 - beta_1) * grad
        v = beta_2 * v + (1 - beta_2) * np.square(grad - m) + epsilon
        x -= learning_rate_next * m / (np.sqrt(v) + epsilon)

        if np.all(np.abs(grad) < epsilon):
            break


    eigen_val = vibration_analysis(function.evaluate, x, eigen_check)

    return history, grad, eigen_val


def conjugate_gradient(function, x, learning_rate, epsilon, max_iteration, eigen_check):
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

    eigen_val = vibration_analysis(function.evaluate, x, eigen_check)

    return history, grad, eigen_val


def newton_method(function, x, learning_rate, epsilon, max_iteration, eigen_check):
    history = []
    for itr in range(max_iteration):
        grad = get_gradient(function.evaluate, x)
        hessian = get_hessian(function.evaluate, x)

        grad_norm = np.sum(grad ** 2) ** 0.5
        history.append([x[0], x[1], function.evaluate(x), grad_norm])

        x -= np.linalg.inv(hessian) @ grad

        if np.all(np.abs(grad) < epsilon):
            break

    eigen_val = vibration_analysis(function.evaluate, x, eigen_check)

    return history, grad, eigen_val


class Optimizer:
    def __init__(self, optimizer, learning_rate=0.01, epsilon=1e-5, max_iteration=30000, eigen_check=True):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iteration = max_iteration
        self.eigen_check = eigen_check

    def optimize(self, function):
        initial_point = function.generate_initial_point()
        history, grad, eigen_val = self.optimizer(function, initial_point, self.learning_rate, self.epsilon, self.max_iteration, self.eigen_check)
        self.check_convengence(grad, eigen_val)

        return history
    
    def check_convengence(self, grad, eigen_val):
        print('\nOptimization results')
        print('Gradient (df/dx):', grad[0])
        print('Gradirnt (df/dy):', grad[1])
        print('Eigenvalue 1    :', eigen_val[0])
        print('Eigenvalue 2    :', eigen_val[1])
        print('')

        if np.all(np.abs(grad) < self.epsilon):
            print('Succeeded to find local minimum or maximum')
        else:
            print('Falied to find local minimum or maximum')

        if not self.eigen_check:
            print('Eigenvalue checks were skipped.')
        elif np.all(eigen_val > 0):
            print('Minimum point was found')
        elif (eigen_val[0] > 0 and eigen_val[1] < 0) or (eigen_val[0] < 0 and eigen_val[1] > 0):
            print('1st-order saddle point was found')
        else:
            print('Converged on an inappropriate point')