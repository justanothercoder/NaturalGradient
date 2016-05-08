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

import autoencoder
import deep

def main():
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

#    n_epochs = 300
    n_pretrain_epochs=300
    n_epochs = 100
    n_hidden = 300

    objective = lasagne.objectives.binary_crossentropy

    models = {
        'adam_sparse': (lasagne.updates.momentum, {'learning_rate': 1.0, 'momentum': 0.9})
    }

    for model in models.keys():
        update, update_params = models[model]

#        network = autoencoder.DenoisingAutoEncoder(n_input=X_train.shape[1], n_hidden=n_hidden)
        network = deep.DeepAutoEncoder(n_input=X_train.shape[1], hidden_sizes=[300, 2])

#        train_err, val_err = network.train(X_train, X_val, 
#                                           n_epochs=n_epochs, n_pretrain_epochs=n_pretrain_epochs,
#                                           objective=objective, update=update, **update_params)

        network.pretrain(X_train, X_val, n_epochs=n_pretrain_epochs, objective=objective, update=update, **update_params)
        network.finish_network()

        np.savez('models/model_%s_pretrained.npz' % model, *lasagne.layers.get_all_param_values(network.output_layer))

        train_err, val_err = network.train(X_train, X_val, 
                                           n_epochs=n_epochs, n_pretrain_epochs=n_pretrain_epochs,
                                           objective=objective, update=update, **update_params)

        plt.plot(val_err, label=model)
    
        np.savez('models/model_%s.npz' % model, *lasagne.layers.get_all_param_values(network.output_layer))
        np.savez('models/model_%s_val_error.npz' % model, val_err)

    plt.title('Validation error/epoch')    
    plt.legend()
    plt.show()
        

if __name__ == '__main__':
    main()
