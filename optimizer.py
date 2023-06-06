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

    eigen_val, eigen_vec = vibration_analysis(function.evaluate, x, eigen_check)

    return history, grad, eigen_val, eigen_vec


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

    eigen_val, eigen_vec = vibration_analysis(function.evaluate, x, eigen_check)

    return history, grad, eigen_val, eigen_vec

    
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

    eigen_val, eigen_vec = vibration_analysis(function.evaluate, x, eigen_check)

    return history, grad, eigen_val, eigen_vec


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

    eigen_val, eigen_vec = vibration_analysis(function.evaluate, x, eigen_check)

    return history, grad, eigen_val, eigen_vec


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

    eigen_val, eigen_vec = vibration_analysis(function.evaluate, x, eigen_check)
       
    return history, grad, eigen_val, eigen_vec


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

    eigen_val, eigen_vec = vibration_analysis(function.evaluate, x, eigen_check)

    return history, grad, eigen_val, eigen_vec


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

    eigen_val, eigen_vec = vibration_analysis(function.evaluate, x, eigen_check)

    return history, grad, eigen_val, eigen_vec


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

    eigen_val, eigen_vec = vibration_analysis(function.evaluate, x, eigen_check)

    return history, grad, eigen_val, eigen_vec


class Optimizer:
    def __init__(self, optimizer=adam, second_optimizer=adam, learning_rate=0.01, epsilon=1e-5, max_iteration=30000, eigen_check=True, find_mep=True):
        self.optimizer = optimizer
        self.second_optimizer = second_optimizer
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iteration = max_iteration
        self.eigen_check = eigen_check
        self.find_mep = find_mep

    def optimize(self, function):
        initial_point = function.generate_initial_point()
        history, grad, eigen_val, eigen_vec = self.optimizer(function, initial_point, self.learning_rate, self.epsilon, self.max_iteration, self.eigen_check)

        opt_history = [history]
        ans = self.check_convengence(grad, eigen_val)

        # ans=-1: fail, ans=0: skip analysis, ans=1: local minimum, ans=2:saddle point
        if ans == 2 and self.find_mep:
            saddle_x = np.array(history[-1][:2])
            mep_history = self.get_mep(function, saddle_x, eigen_val, eigen_vec)
            opt_history += mep_history

        return opt_history
    
    def get_mep(self, function, saddle_x, eigen_val, eigen_vec, alpha=1e-2):
        # 負の固有値の固有ベクトルを取得
        for val, vec in zip(eigen_val, eigen_vec):
            if val < 0:
                negative_vec = vec
                break

        print('The options \'find_mep\' was set to True')
        print('Start searching for minimum energy path from the saddle point\n')

        history = []
        opt_mode = ['Forward search', 'Backward search']
        vec_list = [negative_vec, -negative_vec]
        for mode, vec in zip(opt_mode, vec_list):
            print(mode)

            delta = vec * alpha
            initial_point = saddle_x + delta

            history_mep, grad_mep, eigen_val_mep, eigen_vec_mep = self.second_optimizer(function, initial_point, self.learning_rate, self.epsilon, self.max_iteration, self.eigen_check)
            ans_mep = self.check_convengence(grad_mep, eigen_val_mep)

            history.append(history_mep)

        return history
    
    def check_convengence(self, grad, eigen_val):
        print('\n* * * Optimization results * * *')
        print('Gradient (df/dx):', grad[0])
        print('Gradirnt (df/dy):', grad[1])
        print('Eigenvalue 1    :', eigen_val[0])
        print('Eigenvalue 2    :', eigen_val[1])
        print('')

        if not np.all(np.abs(grad) < self.epsilon):
            print('Falied to find local minimum or saddle point')
            ans = -1
        elif not self.eigen_check:
            print('Eigenvalue checks were skipped.')
            ans = 0
        elif np.all(eigen_val > 0):
            print('Minimum point was found')
            ans = 1
        elif (eigen_val[0] > 0 and eigen_val[1] < 0) or (eigen_val[0] < 0 and eigen_val[1] > 0):
            print('1st-order saddle point was found')
            ans = 2
        else:
            print('Converged on an inappropriate point')
            ans = -1

        print('')

        return ans