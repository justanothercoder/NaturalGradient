import numpy as np

import theano
import theano.tensor as T

import lasagne

from neuralnet import train
from custom_updates import *

from SVRGOptimizer import SVRGOptimizer

def autoencoder_network(input_var, n_input, n_hidden, 
                        W=lasagne.init.GlorotUniform(), 
                        bhid=lasagne.init.Constant(0.), bvis=lasagne.init.Constant(0.)):

    input_layer  = lasagne.layers.InputLayer(shape=(None, n_input), input_var=input_var)
    hidden_layer = lasagne.layers.DenseLayer(input_layer, num_units=n_hidden, nonlinearity=lasagne.nonlinearities.sigmoid, W=W, b=bhid)
    output_layer = lasagne.layers.DenseLayer(hidden_layer, num_units=n_input, nonlinearity=lasagne.nonlinearities.sigmoid, W=hidden_layer.W.T, b=bvis)

    return input_layer, hidden_layer, output_layer

class DenoisingAutoEncoder:
    def __init__(self, n_input, n_hidden, input_var=None, 
                 W=lasagne.init.GlorotUniform(), bvis=lasagne.init.Constant(0.), bhid=lasagne.init.Constant(0.)):
        self.n_input  = n_input
        self.n_hidden = n_hidden

        self.input_var = input_var or T.matrix('inputs')
        self.target_var = T.matrix('targets')
   
        self.input_layer, self.hidden_layer, self.output_layer = autoencoder_network(self.input_var, self.n_input, self.n_hidden, W=W, bhid=bhid, bvis=bvis)
        
        self.hidden_output = lasagne.layers.get_output(self.hidden_layer)
    
    def train(self, X_train, X_val=None, 
            objective=lasagne.objectives.binary_crossentropy, 
            update=lasagne.updates.adam, 
            n_epochs=100, batch_size=500, lambd=0.0,
            **update_params):

        network = self.output_layer

        prediction = lasagne.layers.get_output(network, self.input_var)

        l2_reg = lasagne.regularization.regularize_layer_params(network, lasagne.regularization.l2)
        loss = objective(prediction, self.target_var) + lambd * l2_reg
        loss = loss.mean()
    
        params = lasagne.layers.get_all_params(network, trainable=True)

        svrg = (update == custom_svrg1)
        
        if svrg:
            optimizer = SVRGOptimizer(update_params['m'], update_params['learning_rate'], update_params.get('adaptive', True))
            train_error, validation_error = optimizer.minimize(loss, params, 
                    np.array(X_train * np.random.binomial(size=X_train.shape, n=1, p=0.8), dtype=np.float32), X_train, 
                    self.input_var, self.target_var, X_val, X_val, n_epochs=n_epochs, batch_size=batch_size)
        else:
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
                    np.array(X_train * np.random.binomial(size=X_train.shape, n=1, p=0.8), dtype=np.float32), X_train,
                    X_val, X_val,
                    train_fn, val_fn,
                    n_epochs,
                    batch_size=batch_size#, toprint=it
            )

        return train_error, validation_error

    def get_hidden(self, X):
        return lasagne.layers.get_output(self.hidden_layer, X).eval()
