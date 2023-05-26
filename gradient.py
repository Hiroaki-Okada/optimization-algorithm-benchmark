import pdb

import numpy as np


def get_gradient(f, x):
    dx = _dx(f, x)
    dy = _dy(f, x)

    grad = np.array([dx, dy])

    return grad


def get_hessian(f, x):
    dxx = _dxx(f, x)
    dxy = _dxy(f, x)
    dyx = _dxy(f, x)
    dyy = _dyy(f, x)

    hessian = np.array([[dxx, dxy],
                        [dyx, dyy]])
    
    return hessian


def _dx(f, x, h=1e-4):
    x1, x2 = x
    return (f(np.array([x1 + h, x2])) - f(np.array([x1 - h, x2]))) / (2 * h)


def _dy(f, x, h=1e-4):
    x1, x2 = x
    return (f(np.array([x1, x2 + h])) - f(np.array([x1, x2 - h]))) / (2 * h)


def _dxx(f, x, h=1e-4):
    x1, x2 = x
    return (_dx(f, np.array([x1 + h, x2])) - _dx(f, np.array([x1 - h, x2]))) / (2 * h)


def _dxy(f, x, h=1e-4):
    x1, x2 = x
    return (_dx(f, np.array([x1, x2 + h])) - _dx(f, np.array([x1, x2 - h]))) / (2 * h)


def _dyy(f, x, h=1e-4):
    x1, x2 = x
    return (_dy(f, np.array([x1, x2 + h])) - _dy(f, np.array([x1, x2 - h]))) / (2 * h)