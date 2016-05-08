import numpy as np

import theano
import theano.tensor as T

import lasagne

from neuralnet import train
from autoencoder import autoencoder_network

class SparseAutoEncoder:
    def __init__(self, n_input, n_hidden, input_var=None, 
                 W=lasagne.init.GlorotUniform(), bvis=lasagne.init.Constant(0.), bhid=lasagne.init.Constant(0.)):
        self.n_input  = n_input
        self.n_hidden = n_hidden

        self.input_var = input_var or T.matrix('inputs')
        self.target_var = T.matrix('targets')
   
        i, h, o = autoencoder_network(self.input_var, self.n_input, self.n_hidden, W=W, bhid=bhid, bvis=bvis)
        self.input_layer, self.hidden_layer, self.output_layer = i, h, o 
        
        self.hidden_output = lasagne.layers.get_output(self.hidden_layer)
    
    def train(self, X_train, X_val=None, 
            objective=lasagne.objectives.binary_crossentropy, 
            update=lasagne.updates.adam, 
            n_epochs=100, batch_size=500, rho=0.05, beta=0.1, lambd = 1.0,
            **update_params):

        network = self.output_layer

        prediction = lasagne.layers.get_output(network)        
        loss = objective(prediction, self.target_var)

        sparsity_loss = lasagne.objectives.binary_crossentropy(lasagne.layers.get_output(self.hidden_layer).mean(axis=0), T.extra_ops.repeat(T.constant(rho), self.n_hidden)).sum()

        loss = loss + beta * sparsity_loss + lambd * lasagne.regularization.regularize_layer_params(network, lasagne.regularization.l2)

        loss = loss.mean()
    
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = update(loss, params, **update_params)

        train_fn = theano.function([self.input_var, self.target_var], loss, updates=updates)
    
        if X_val is not None:
            test_prediction = lasagne.layers.get_output(network, deterministic=True)
            test_loss = objective(test_prediction, self.target_var)
            test_loss = test_loss.mean()
            val_fn = theano.function([self.input_var, self.target_var], test_loss)
        else:
            val_fn = None
        
        train_error, validation_error = train(
                X_train, X_train,
                X_val, X_val,
                train_fn, val_fn,
                n_epochs,
                batch_size=batch_size,
                network=self
        )

        return train_error, validation_error

    def get_hidden(self, X):
        return lasagne.layers.get_output(self.hidden_layer, X).eval()
