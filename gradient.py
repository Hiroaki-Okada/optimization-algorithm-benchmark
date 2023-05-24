
def central_difference(f, x, y, h=1e-4):
    grad_x = (f(x + h, y) - f(x - h, y)) / (2 * h)
    grad_y = (f(x, y + h) - f(x, y - h)) / (2 * h)

    return grad_x, grad_y