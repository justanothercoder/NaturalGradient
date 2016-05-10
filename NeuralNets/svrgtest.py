import sys, os, time
import numpy as np

import theano
import theano.tensor as T

import lasagne

import matplotlib.pyplot as plt
import seaborn

from neuralnet import get_network_stats
from custom_updates import *
import custom_objectives

from load_dataset import *

import autoencoder
import deep

def main():
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    X_train = X_train[:]

    n_epochs = 100
    n_hidden = 300

    objective = lasagne.objectives.binary_crossentropy
    #objective = lasagne.objectives.squared_error

    models = {
    #    'svrg_test_100': (custom_svrg, {'learning_rate': 100.0, 'm': 500})
    #    'momentum_1.0_0.9_300': (custom_momentum, {'learning_rate': 1.0, 'momentum': 0.9}),
    #    'adam_test_100': (custom_adam, {'learning_rate': 0.01}),        
        'svrg_wrong': (custom_svrg1, {'learning_rate': 100.0, 'm': 500})
    }

    for model in models.keys():
        update, update_params = models[model]

        network = autoencoder.DenoisingAutoEncoder(n_input=X_train.shape[1], n_hidden=n_hidden)

        train_err, val_err = network.train(X_train, X_val, n_epochs=n_epochs, batch_size=500, lambd=0.0,
                                           objective=objective, update=update, **update_params)

        plt.plot(val_err, label=model)
    
        np.savez('models/model_%s.npz' % model, *lasagne.layers.get_all_param_values(network.output_layer))
        np.savez('models/model_%s_val_error.npz' % model, val_err)

    plt.title('Validation error/epoch')    
    plt.legend()
    plt.show()
        

if __name__ == '__main__':
    main()
