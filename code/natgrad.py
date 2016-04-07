import numpy as np

def natural_gradient_descent(x0, G_inv, grad, alpha, eps=0.0001, f=None, iters=None):
    x = x0.copy()
    d = np.dot(G_inv(x), grad(x, 0))

    points = [x.copy()]

    hist = None
    if f is not None:
        hist = [f(x, 0)]
    
    iter_num = 1

    if iters is None:
        is_end = lambda : np.sum(d**2) < eps
    else:
        is_end = lambda : iter_num >= iters

    while not is_end():
        x -= alpha * d
        d = np.dot(G_inv(x), grad(x))

        points.append(x.copy())

        if f is not None:
            hist.append(f(x))

        iter_num += 1

    points = np.array(points)
    if f is not None:
        return x, points, hist
    return x, points

def simple_gradient_descent(x0, grad, alpha, eps=0.0001, f=None, iters=None):
    return natural_gradient_descent(x0, lambda x: np.eye(x0.shape[0]), grad, alpha, eps, f, iters)

