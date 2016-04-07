import numpy as np
import matplotlib.pyplot as plt
import math
from natgrad import *

from polar_test import polar_G

def test():

    r_opt = 1.0

    hist_av = np.zeros(200)
    hist_av_simple = np.zeros(200)
    
    for i in range(100):

        phi_opt = np.random.uniform(0.0, math.pi)
        
        r0 = np.random.uniform(0.0, 1.0)
        phi0 = np.random.uniform(0.0, math.pi)

        sin_x0 = np.array([r0, phi0])
        polar_G_inv = lambda w: np.linalg.pinv(polar_G(w))
        sin_alpha = 0.2

        iters = 200

        omega = 0.5

        sigma = 0.0001
        y = np.array([r_opt * math.cos(omega * k + phi_opt) + np.random.randn() * sigma for k in range(iters)])

        def sin_f(w, k):
            r, phi = w[0], w[1]
            return (y[k] - r * math.cos(omega * k + phi))**2

        def sin_f_grad(w, k):
            r, phi = w[0], w[1]
            arg = omega * k + phi
            return np.array([
                    (y[k] - r * math.cos(arg)) * (-math.cos(arg)),
                    (y[k] - r * math.cos(arg)) * r * math.sin(arg)
                ])

        polar_x, _, hist = natural_gradient_descent(sin_x0, polar_G_inv, sin_f_grad, sin_alpha, f=sin_f, iters=iters)
        polar_x_simple, _, hist_simple = simple_gradient_descent(sin_x0, sin_f_grad, sin_alpha, f=sin_f, iters=iters)

        hist_av += np.array(hist)
        hist_av_simple += np.array(hist_simple)

    hist_av /= 1000.0
    hist_av_simple /= 1000.0
    
    plt.plot(hist_av, color='b', label="Natural gradient")
    plt.plot(hist_av_simple, color='r', label="Simple gradient")
    plt.legend()
    plt.show()
