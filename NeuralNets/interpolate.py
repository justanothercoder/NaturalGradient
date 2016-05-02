import sys, os

import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as T 
import lasagne

from load_dataset import *
from neuralnet import build

import seaborn

def interpolate(x1, x2, nums=10):
    d = (x2 - x1) / float(nums)

    return np.array([x1 + i * d for i in range(nums + 1)])

def main():
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')
        
    model = 'adam_reg'
    with np.load('models/model_%s.npz' % model) as f:
        param_values = [f['arr_%d' % j] for j in range(len(f.files))]
    
        encoder = lasagne.layers.DenseLayer(
                lasagne.layers.InputLayer(shape=(None, 784), input_var=input_var), 
                num_units=300, 
                nonlinearity=lasagne.nonlinearities.sigmoid
        )

        lasagne.layers.set_all_param_values(encoder, param_values[:2])

        decoder = lasagne.layers.DenseLayer(
                lasagne.layers.InputLayer(shape=(None, 300), input_var=input_var), 
                num_units=784, 
                nonlinearity=lasagne.nonlinearities.sigmoid
        )
        
        lasagne.layers.set_all_param_values(decoder, param_values[2:])

    n_images = 10
    n_rows = 10

    for row in range(n_rows):

        n1 = np.random.randint(0, X_train.shape[0])
        n2 = np.random.randint(0, X_train.shape[0])

        enc1 = lasagne.layers.get_output(encoder, X_train[n1]).eval()
        enc2 = lasagne.layers.get_output(encoder, X_train[n2]).eval()

        images = lasagne.layers.get_output(decoder, interpolate(enc1, enc2, n_images)).eval()
        
        for i in range(n_images):
            plt.subplot(n_rows, n_images, row * n_images + i + 1)
            plt.axis('off')
            plt.imshow(images[i].reshape(28, 28), cmap='Greys')

    plt.show()

main()
