import numpy as np

import theano
import theano.tensor as T

import lasagne

from neuralnet import train

from SVRGOptimizer import SVRGOptimizer

def classifier_network(input_var, n_input, n_hidden, n_output):

    input_layer  = lasagne.layers.InputLayer(shape=(None, n_input), input_var=input_var)
    hidden_layer = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(input_layer, p=0.5), 
            num_units=n_hidden, 
            nonlinearity=lasagne.nonlinearities.sigmoid)
    output_layer = lasagne.layers.DenseLayer(hidden_layer, num_units=n_output, nonlinearity=lasagne.nonlinearities.softmax)

    return input_layer, hidden_layer, output_layer

class NeuralClassifier:
    def __init__(self, n_input, n_hidden, n_output, input_var=None):
        self.n_input  = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.input_var = input_var or T.matrix('inputs')
        self.target_var = T.ivector('targets')
   
        self.input_layer, self.hidden_layer, self.output_layer = classifier_network(self.input_var, n_input, n_hidden, n_output)
    
    def train(self, X_train, Y_train, X_val=None, Y_val=None,
            objective=lasagne.objectives.binary_crossentropy, 
            update=lasagne.updates.adam, 
            n_epochs=100, batch_size=500, lambd=0.0,
            **update_params):

        network = self.output_layer

        prediction = lasagne.layers.get_output(network)

        l2_reg = lasagne.regularization.regularize_layer_params(network, lasagne.regularization.l2)
        loss = objective(prediction, self.target_var) + lambd * l2_reg
        loss = loss.mean()
    
        params = lasagne.layers.get_all_params(network, trainable=True)

        svrg = False
  #      svrg = True
        
        if svrg:
            optimizer = SVRGOptimizer(update_params['m'], update_params['learning_rate'])
            train_error, validation_error = optimizer.minimize(loss, params, 
                    X_train, Y_train, 
                    self.input_var, self.target_var, 
                    X_val, Y_val, 
                    n_epochs=n_epochs)
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
                    X_train, Y_train, X_val, Y_val,
                    train_fn, val_fn,
                    n_epochs, batch_size=batch_size#, toprint=it
            )

        return train_error, validation_error
