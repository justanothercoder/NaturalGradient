import numpy as np
import matplotlib.pyplot as plt
import math
from natgrad import *
from pprint import pprint

def polar_G(w):        
    return np.array([
        [1.0, 0.0],
        [0.0, w[0]**2]
    ])

def polar_f(w, k=None):
    r, phi = w[0], w[1]
    return (r * math.cos(phi) - 1)**2 + (r * math.sin(phi))**2

def polar_f_grad(w, k=None):
    r, phi = w[0], w[1]
    return np.array([
            2 * (r - math.cos(phi)),
            2 * r * math.sin(phi)
        ])

def polar_to_cartesian(r_phi):
    x = np.zeros(r_phi.shape)
    r, phi = r_phi[:, 0], r_phi[:, 1]

    x[:, 0] = r * np.cos(phi)
    x[:, 1] = r * np.sin(phi)

    return x


def test():

    polar_x0 = np.array([0.7, 3 * math.pi / 4.0])
    polar_G_inv = lambda w: np.linalg.pinv(polar_G(w))
    polar_alpha = 0.01

    polar_x, polar_points, hist = natural_gradient_descent(polar_x0, polar_G_inv, polar_f_grad, polar_alpha, f=polar_f)
    polar_x_simple, polar_points_simple, hist_simple = simple_gradient_descent(polar_x0, polar_f_grad, polar_alpha, f=polar_f)
    
    plt.plot(hist, color='b', label="Natural gradient")
    plt.plot(hist_simple, color='r', label="Simple gradient")
    plt.legend()
    plt.show()

    polar_points = polar_to_cartesian(polar_points)
    polar_points_simple = polar_to_cartesian(polar_points_simple)

    print "Iterations (natural): ", len(polar_points)
    print "Iterations (simple): ", len(polar_points_simple)

#    plt.plot(polar_points[:, 0], polar_points[:, 1], color='b', label="Natural gradient", linestyle='-', marker='o')
#    plt.plot(polar_points_simple[:, 0], polar_points_simple[:, 1], color='r', label="Simple gradient", linestyle='--', marker='o')

    plt.plot(polar_points[:, 0], polar_points[:, 1], color='b', label="Natural gradient", linestyle=':')
    plt.plot(polar_points_simple[:, 0], polar_points_simple[:, 1], color='r', label="Simple gradient", linestyle='--')

    plt.legend()
    plt.show()
