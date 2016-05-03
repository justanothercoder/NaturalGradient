import numpy as np

import theano
import theano.tensor as T

import lasagne

from neuralnet import train
import autoencoder

class DeepAutoEncoder:
    def __init__(self, n_input, hidden_sizes, input_var=None):
        self.n_layers = len(hidden_sizes)
        self.n_input = n_input

        self.hidden_layers = []
        self.dA_layers = []

        self.input_var = input_var or T.matrix('inputs')
        self.target_var = T.matrix('targets')

        self.hidden_layers.append(lasagne.layers.InputLayer(shape=(None, n_input), input_var=self.input_var))

        inp_var = self.input_var
        in_size = n_input        
       
        for i in range(self.n_layers):
            hid = lasagne.layers.DenseLayer(self.hidden_layers[-1], num_units=hidden_sizes[i], nonlinearity=lasagne.nonlinearities.sigmoid)
            dA = autoencoder.DenoisingAutoEncoder(in_size, hidden_sizes[i], input_var=inp_var, W=hid.W, bhid=hid.b)

            inp_var = lasagne.layers.get_output(self.hidden_layers[-1])
            in_size = hidden_sizes[i]
            
            self.hidden_layers.append(hid)
            self.dA_layers.append(dA)

        out = lasagne.layers.DenseLayer(self.hidden_layers[-1], num_units=n_input, nonlinearity=lasagne.nonlinearities.sigmoid)
        self.hidden_layers.append(out)

        self.output_layer = out

    def train(self, X_train, X_val=None,
            objective=lasagne.objectives.binary_crossentropy, 
            update=lasagne.updates.adam, 
            n_epochs=100, batch_size=500,
            **update_params):

        pretrain_X = X_train

        print ("Pretraining...")
        for i, dA in enumerate(self.dA_layers):
            print ("Pretraining {}->{} layers".format(i, i+1))
            dA.train(pretrain_X, None, objective=objective, update=update, n_epochs=n_epochs, batch_size=batch_size)
            pretrain_X = dA.get_hidden(pretrain_X)
        print ("Pretraining finished")

        network = self.hidden_layers[-1]

        prediction = lasagne.layers.get_output(network)
        loss = objective(prediction, self.target_var)
        loss = loss.mean()
    
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = update(loss, params, **update_params)

        train_fn = theano.function([self.input_var, self.target_var], loss, updates=updates)
    
        if X_val is not None:
            test_prediction = lasagne.layers.get_output(network, deterministic=True)
            test_loss = objective(test_prediction, self.target_var)
            test_loss = test_loss.mean()
            val_fn = theano.function([self.input_var, self.target_var], test_loss)
        
        train_error, validation_error = train(
                X_train, X_train, X_val, X_val,
                train_fn, val_fn,
                n_epochs, batch_size=batch_size
        )

        return train_error, validation_error
