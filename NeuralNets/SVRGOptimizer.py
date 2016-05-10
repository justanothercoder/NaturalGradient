import numpy as np

import theano
import theano.tensor as T

from theano.ifelse import ifelse

from neuralnet import iterate_minibatches

from collections import OrderedDict
import time

class SVRGOptimizer:
    def __init__(self, m, learning_rate):
        self.m = m
        self.learning_rate = learning_rate

    def minimize(self, loss, params, X_train, Y_train, input_var, target_var, X_val, Y_val, n_epochs=100):
        self.input_var = input_var
        self.target_var = target_var
        w_updates, mu_updates = self.make_updates(loss, params)

        train_mu = theano.function([self.input_var, self.target_var], loss, updates=mu_updates)
        train_w = theano.function([self.input_var, self.target_var], loss, updates=w_updates)

        val_fn = theano.function([self.input_var, self.target_var], loss)

        train_error = []
        validation_error = []

        batch_size = 500
        num_batches = X_train.shape[0] / batch_size
        n = num_batches

        print("Starting training...")
        for epoch in range(n_epochs):

            t = time.time()
            
            for mu in self.mu:
                mu.set_value(0 * mu.get_value())

            for batch in iterate_minibatches(X_train, Y_train, batch_size, shuffle=True):
                inputs, targets = batch
                train_mu(inputs, targets)
            
            for mu in self.mu:
                mu.set_value(mu.get_value() / n)
            
            train_err = 0
            train_batches = 0

            j = 0
            while j < self.m:
                for batch in iterate_minibatches(X_train, Y_train, batch_size, shuffle=True):
#                    print "Iter: ", self.it_num.get_value()
                    if j >= self.m:
                        break
                    j += 1
                    inputs, targets = batch
                    train_err += train_w(inputs, targets)
                    train_batches += 1
            
            val_err = 0
            val_batches = 0
            for i, batch in enumerate(iterate_minibatches(X_val, Y_val, batch_size, shuffle=True)):
                inputs, targets = batch
                val_err += val_fn(inputs, targets)
                val_batches += 1

            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, n_epochs, time.time() - t))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

            train_error.append(train_err / train_batches)
            validation_error.append(val_err / val_batches)
#            if X_val is not None:
#                print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

        return train_error, validation_error

    def make_updates(self, loss, params):

        mu_updates = self.make_mu_updates(loss, params)
        w_updates = self.make_w_updates(loss, params)
   
        return w_updates, mu_updates

    def make_mu_updates(self, loss, params):
        mu_updates = OrderedDict()

        grads = theano.grad(loss, params)

        self.mu = []
        for param, grad in zip(params, grads):
            value = param.get_value(borrow=True)

            mu = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            mu_updates[mu] = mu + grad
            self.mu.append(mu)

        return mu_updates

    def make_w_updates(self, loss, params):
        w_updates = OrderedDict()
        
        params_tilde = [theano.shared(x.get_value()) for x in params] 
        loss_tilde = theano.clone(loss, replace=zip(params, params_tilde))

        grads = theano.grad(loss, params)
        grads_tilde = theano.grad(loss_tilde, params_tilde)

        it_num = theano.shared(np.cast['int16'](0))
        it = it_num + 1

        for param, grad, mu, param_tilde, grad_tilde in zip(params, grads, self.mu, params_tilde, grads_tilde):
            new_param = param - self.learning_rate * (grad - grad_tilde + mu)
            w_updates[param] = new_param
            w_updates[param_tilde] = ifelse(T.eq(it, self.m), new_param, param_tilde)

        self.it_num = it_num
        
        w_updates[it_num] = ifelse(T.eq(it, self.m), np.cast['int16'](0), it)
        return w_updates
