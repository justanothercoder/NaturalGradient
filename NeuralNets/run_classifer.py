import sys, os

import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as T 
import lasagne

from load_dataset import *
from neuralnet import build

from deep import DeepAutoEncoder
from sparse_autoencoder import SparseAutoEncoder
from neuralclassifier import NeuralClassifier

def main():
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')

    n_hiddens = [300, 300, 500]
    methods = ['adam_classif', 'adam_classif_dropout', 'adam_classif_on_sparse']

    for (i, model), n_hidden in zip(enumerate(methods), n_hiddens):
        network = NeuralClassifier(784, n_hidden, 10).output_layer
        with np.load('models/model_%s.npz' % model) as f:
            param_values = [f['arr_%d' % j] for j in range(len(f.files))]
            lasagne.layers.set_all_param_values(network, param_values)

        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        
        accuracy = (lasagne.layers.get_output(network, X_test, deterministic=True).eval().argmax(axis=1) == y_test).mean()
        print "Model: {}; accuracy: {}".format(model, accuracy)

main()
