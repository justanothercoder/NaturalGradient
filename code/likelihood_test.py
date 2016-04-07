import numpy as np
import matplotlib.pyplot as plt
import math
from natgrad import *

def test():

    n = 1000
    X = np.random.randn(n)

    x0 = 10 * np.random.randn(2)
    x0[1] = np.abs(x0[1])

    print "True mean: ", x0[0]
    print "True variance: ", x0[1]

    def loglikelihood(w, k=None):
        a, sigma = w[0], w[1]
        return sum(((x - a)**2 / (2 * sigma**2) for x in X)) + n * np.log(sigma) + n / 2.0 * np.log(2.0 * math.pi)

    def loglikelihood_grad(w, k=None):
        a, sigma = w[0], w[1]
        print "Mean: ", a, "Variance: ", sigma
        return np.array([
            -1.0 / sigma**2 * sum((x - a for x in X)),
            -1.0 / sigma**3 * sum(((x - a)**2 for x in X)) + n / sigma
        ])

    def G(w):
        a, sigma = w[0], w[1]
        G = np.zeros((2, 2))
        for x in X:
            log_grad = np.array([
                [(x - a) / (2 * sigma ** 2)],
                [(x - a)**2 / sigma ** 3 - 1.0 / sigma]
            ])
            G += np.dot(log_grad, log_grad.T)
        return G / n

    G_inv = lambda w: np.linalg.pinv(G(w))

    alpha = 0.000005

    x, points, hist = natural_gradient_descent(x0, G_inv, loglikelihood_grad, alpha, f=loglikelihood)
    x_simple, points_simple, hist_simple = simple_gradient_descent(x0, loglikelihood_grad, alpha, f=loglikelihood)

    plt.plot(hist, color='b', label="Natural gradient")
    plt.plot(hist_simple, color='r', label="Simple gradient")
    plt.legend()
    plt.show()

    plt.plot(points[:, 0], points[:, 1], color='b', label="Natural gradient", linestyle=":")
    plt.plot(points_simple[:, 0], points_simple[:, 1], color='r', label="Simple gradient", linestyle="--")

    plt.legend()
    plt.show()

