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

def main():
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    n_epochs = 300
    n_hidden = 500

    objective = lasagne.objectives.binary_crossentropy

    models = {
#        'adam': (lasagne.updates.adam, {'learning_rate': 0.01}),
#        'momentum': (lasagne.updates.momentum, {'learning_rate': 0.01, 'momentum': 0.9}),
#        'nesterov_momentum': (lasagne.updates.nesterov_momentum, {'learning_rate': 0.01, 'momentum': 0.9}),
#        'adagrad': (lasagne.updates.adagrad, {'learning_rate': 0.01}),
#        'rmsprop': (lasagne.updates.rmsprop, {'learning_rate': 0.01, 'rho': 0.9})

#        'custom_momentum-1.0-0.9': (custom_momentum, {'learning_rate': 1.0, 'momentum': 0.9}),
#        'custom_momentum-0.1-0.9': (custom_momentum, {'learning_rate': 0.1, 'momentum': 0.9}),
#        'custom_momentum-0.01-0.9': (custom_momentum, {'learning_rate': 0.01, 'momentum': 0.9}),
#        'custom_momentum-0.001-0.9': (custom_momentum, {'learning_rate': 0.001, 'momentum': 0.9}),
#        'custom_momentum-0.1-0.5': (custom_momentum, {'learning_rate': 0.1, 'momentum': 0.5}),
#        'custom_momentum-0.1-0.1': (custom_momentum, {'learning_rate': 0.1, 'momentum': 0.1})

#        'custom_momentum-1.0divk**1.0-0.9': (custom_momentum, {'learning_rate': 1.0, 'momentum': 0.9, 'tau': 1.0}),
#        'custom_momentum-1.0divk**0.75-0.9': (custom_momentum, {'learning_rate': 1.0, 'momentum': 0.9, 'tau': 0.75}),
#        'custom_momentum-1.0divk**0.5-0.9': (custom_momentum, {'learning_rate': 1.0, 'momentum': 0.9, 'tau': 0.5}),

#        'custom_nesterov_momentum-1.0divk**1.0-0.9': (custom_nesterov_momentum, {'learning_rate': 1.0, 'momentum': 0.9, 'tau': 1.0}),
#        'custom_nesterov_momentum-1.0divk**0.75-0.5': (custom_nesterov_momentum, {'learning_rate': 1.0, 'momentum': 0.5, 'tau': 0.75}),
#        'custom_nesterov_momentum-1.0divk**0.5-0.1': (custom_nesterov_momentum, {'learning_rate': 1.0, 'momentum': 0.1, 'tau': 0.5}),
#        'custom_nesterov_momentum-100.0divk**0.5-0.9': (custom_nesterov_momentum, {'learning_rate': 100.0, 'momentum': 0.9, 'tau': 0.5}),

#        'custom_adagrad_10.0': (custom_adagrad, {'learning_rate': 10.0}),
#        'custom_adagrad_1.0': (custom_adagrad, {'learning_rate': 1.0}),
#        'custom_adagrad_0.1': (custom_adagrad, {'learning_rate': 0.1}),
#        'custom_adagrad_0.01': (custom_adagrad, {'learning_rate': 0.01}),

#        'custom_rmsprop_0.01-0.9': (custom_rmsprop, {'learning_rate': 0.01, 'rho': 0.9}),
#        'custom_rmsprop_0.01-0.6': (custom_rmsprop, {'learning_rate': 0.01, 'rho': 0.6}),
#        'custom_rmsprop_0.01-0.3': (custom_rmsprop, {'learning_rate': 0.01, 'rho': 0.3}),
#        'custom_rmsprop_0.01-0.1': (custom_rmsprop, {'learning_rate': 0.01, 'rho': 0.1}),

#        'custom_adam_0.01_0.9_0.999': (custom_adam, {'learning_rate': 0.01, 'beta1': 0.9, 'beta2': 0.999}), # best try
#        'custom_adam_0.01_0.5_0.999': (custom_adam, {'learning_rate': 0.01, 'beta1': 0.5, 'beta2': 0.999}),
#        'custom_adam_0.01_0.1_0.999': (custom_adam, {'learning_rate': 0.01, 'beta1': 0.1, 'beta2': 0.999}),
#
#        'custom_adam_0.01_0.9_0.5': (custom_adam, {'learning_rate': 0.01, 'beta1': 0.9, 'beta2': 0.5}),
#        'custom_adam_0.01_0.9_0.1': (custom_adam, {'learning_rate': 0.01, 'beta1': 0.9, 'beta2': 0.1}),
#        
#        'custom_adam_0.1_0.9_0.999': (custom_adam, {'learning_rate': 0.1, 'beta1': 0.9, 'beta2': 0.999}),
#        'custom_adam_1.0_0.9_0.999': (custom_adam, {'learning_rate': 1.0, 'beta1': 0.9, 'beta2': 0.999}),
#        'custom_adam_10.0_0.9_0.999': (custom_adam, {'learning_rate': 10.0, 'beta1': 0.9, 'beta2': 0.999}),

#        'adam_reg_denoising100': (lasagne.updates.adam, {'learning_rate': 0.01}),
        'momentum_reg_denoising100': (lasagne.updates.momentum, {'learning_rate': 10.0, 'momentum': 0.9}),
    }

    for model in models.keys():
        update, update_params = models[model]
        network, train_err, val_err = get_network_stats(X_train, X_val, X_test, n_epochs, n_hidden, objective, update, **update_params)

        plt.plot(val_err, label=model)
    
        np.savez('models/model_%s.npz' % model, *lasagne.layers.get_all_param_values(network))
        np.savez('models/model_%s_val_error.npz' % model, val_err)

    plt.title('Validation error/epoch')    
    plt.legend()
    plt.show()
        

if __name__ == '__main__':
    main()
