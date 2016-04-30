import sys, os

import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as T 
import lasagne

from load_dataset import load_dataset
from neuralnet import build

import sklearn.manifold

def build_half(input_var=None, n_hidden=300):
    input_layer = lasagne.layers.InputLayer(shape=(None, 784), input_var=input_var)
    hidden_layer = lasagne.layers.DenseLayer(input_layer, num_units=n_hidden, nonlinearity=lasagne.nonlinearities.sigmoid)

    return hidden_layer

def main():
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')

    network = build_half(input_var, n_hidden=300)

    model = 'adam_reg'

    n_images = 10
    
    with np.load('models/model_%s.npz' % model) as f:
        print f.files
        param_values = [f['arr_%d' % j] for j in range(len(f.files[:2]))]
        lasagne.layers.set_all_param_values(network, param_values)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    indices = np.arange(X_test.shape[0])
    np.random.shuffle(indices)
    indices = indices[:3000]

    X, Y = X_test[indices], y_test[indices]

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'darkgreen', 'lime']
            
    out = lasagne.layers.get_output(network, X).eval()
    tsne = sklearn.manifold.TSNE()
    out = tsne.fit_transform(out)

    for j in np.unique(y_test):
        plt.scatter(out[Y == j][:, 0], out[Y == j][:, 1], color=colors[j], label=str(j))

    plt.legend()
    plt.show()

main()
