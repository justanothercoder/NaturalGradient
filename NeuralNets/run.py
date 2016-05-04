import sys, os

import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as T 
import lasagne

from load_dataset import *
from neuralnet import build

from deep import DeepAutoEncoder

def main():
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')

#    network = build(input_var, n_hidden=300)
    network = DeepAutoEncoder(784, [300, 2])
    network.finish_network()
    network = network.output_layer

#    methods = ['adam', 'momentum', 'nesterov_momentum', 'adagrad', 'rmsprop', 'custom_momentum']
#    methods = ['custom_adam_0.01_0.9_0.999', 'adam']
#    methods = ['adam_reg']

    #methods = ['adam_reg_dummy']
#    methods = ['adam_deep300-2-300_0.01']
    methods = ['adam_deep_test_tied']

    n_images = 10
    
    for j in range(n_images):
        plt.subplot(len(methods) + 1, n_images, j + 1)
        #plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        if j == 0:
            plt.ylabel('original', rotation='horizontal')
        plt.imshow(X_train[j].reshape(28, 28), cmap='Greys')

    for i, model in enumerate(methods):
        with np.load('models/model_%s.npz' % model) as f:
            param_values = [f['arr_%d' % j] for j in range(len(f.files))]
            lasagne.layers.set_all_param_values(network, param_values)

        test_prediction = lasagne.layers.get_output(network, deterministic=True)

        for j in range(n_images):
            plt.subplot(len(methods) + 1, n_images, n_images * (i+1) + j + 1)
            #plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            if j == 0:
                plt.ylabel(model, rotation='horizontal')
            plt.imshow(lasagne.layers.get_output(network, X_train[j]).eval().reshape(28, 28), cmap='Greys')

#    n_images = 10
#    for i in range(n_images):
#        plt.subplot(n_images, 2, 2 * i + 1)
#        plt.axis('off')
#        plt.imshow(X_train[i].reshape(28, 28), cmap='Greys')
#        plt.subplot(n_images, 2, 2 * i + 2)
#        plt.axis('off')
#        plt.imshow(lasagne.layers.get_output(network, X_train[i]).eval().reshape(28, 28), cmap='Greys')
    
    plt.show()

main()
