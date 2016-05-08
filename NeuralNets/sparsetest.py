# -*- encoding: utf-8 -*-

import sys, os, time
import numpy as np

import theano
import theano.tensor as T

import lasagne

import matplotlib.pyplot as plt
import seaborn

from neuralnet import get_network_stats
from custom_updates import *

from load_dataset import *

from sparse_autoencoder import SparseAutoEncoder

def main():
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    np.random.shuffle(X_train)
    X_train = X_train[:10000]

    n_epochs = 100
    n_hidden = 2000

    objective = lasagne.objectives.binary_crossentropy

    models = {
            'adam_sparse_7.0_not_denoising': (lasagne.updates.adam, {'learning_rate': 0.01})
    }

    for model in models.keys():
        update, update_params = models[model]

        network = SparseAutoEncoder(n_input=X_train.shape[1], n_hidden=n_hidden)
        train_err, val_err = network.train(X_train, X_val, n_epochs=n_epochs, objective=objective, update=update, beta=0.1, rho=0.01, batch_size=500, lambd=0.000001, **update_params)

#β = 0.01; ρ = 0.01
#β = 0.1; ρ = 0.05

        plt.plot(val_err, label=model)
    
        np.savez('models/model_%s.npz' % model, *lasagne.layers.get_all_param_values(network.output_layer))
        np.savez('models/model_%s_val_error.npz' % model, val_err)

    plt.title('Validation error/epoch')    
    plt.legend()
    plt.show()
        

if __name__ == '__main__':
    main()
