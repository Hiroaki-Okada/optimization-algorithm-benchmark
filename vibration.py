import pdb

import math
import numpy as np
from gradient import get_hessian


def vibration_analysis(f, x, eigen_check, epsilon=1e-6):
    if not eigen_check:
        return np.zeros(2)

    hessian = get_hessian(f, x)

    o_12 = hessian[0][1]
    o_11 = hessian[0][0]
    o_22 = hessian[1][1]

    theta_0 = 0.5 * math.atan((2 * o_12) / (o_11 - o_22 + epsilon))

    eigen_val_1 = o_11 * math.cos(theta_0) ** 2 + o_22 * math.sin(theta_0) ** 2 + o_12 * math.sin(2 * theta_0)
    eigen_val_2 = o_11 * math.sin(theta_0) ** 2 + o_22 * math.cos(theta_0) ** 2 - o_12 * math.sin(2 * theta_0)
    eigen_val = np.array([eigen_val_1, eigen_val_2])

    eigen_vec_1 = [math.cos(theta_0), math.sin(theta_0)]
    eigen_vec_2 = [math.sin(theta_0), -math.cos(theta_0)]
    eigen_vec = np.array([eigen_vec_1, eigen_vec_2])

    eigen_val, eigen_vec = np.linalg.eig(hessian)
    # print(eigen_val)
    # print(eigen_vec)

    return eigen_val