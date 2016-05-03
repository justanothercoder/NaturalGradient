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

def main():
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    n_epochs = 100
    n_hidden = 300

    objective = lasagne.objectives.binary_crossentropy

    models = {
        'adam_reg_dummy': (lasagne.updates.adam, {'learning_rate': 0.01})
    }

    for model in models.keys():
        update, update_params = models[model]

        network = autoencoder.DenoisingAutoEncoder(n_input=X_train.shape[1], n_hidden=n_hidden)
        train_err, val_err = network.train(X_train, X_val, n_epochs=n_epochs, objective=objective, update=update, **update_params)

        plt.plot(val_err, label=model)
    
        np.savez('models/model_%s.npz' % model, *lasagne.layers.get_all_param_values(network.output_layer))
        np.savez('models/model_%s_val_error.npz' % model, val_err)

    plt.title('Validation error/epoch')    
    plt.legend()
    plt.show()
        

if __name__ == '__main__':
    main()
