import numpy as np

import theano
import theano.tensor as T

from theano.ifelse import ifelse

from neuralnet import iterate_minibatches

from collections import OrderedDict
import time

class SVRGOptimizer:
    def __init__(self, m, learning_rate, adaptive=True):
        self.m = m
        self.learning_rate = learning_rate
        self.adaptive = adaptive

        self.counted_gradient = theano.shared(0)

    def minimize(self, loss, params, X_train, Y_train, input_var, target_var, X_val, Y_val, n_epochs=100, batch_size=500):
        self.input_var = input_var
        self.target_var = target_var

        self.L = theano.shared(np.cast['float32'](1. / self.learning_rate))

        w_updates, mu_updates = self.make_updates(loss, params)

        train_mu = theano.function([self.input_var, self.target_var], loss, updates=mu_updates)
        train_w = theano.function([self.input_var, self.target_var], loss, updates=w_updates)

        val_fn = theano.function([self.input_var, self.target_var], loss)

        train_error = []
        validation_error = []

        num_batches = X_train.shape[0] / batch_size
        n = num_batches

        print "NUMBATCHES: ", n

        j = 0

        L_fn = self.make_L_fn(loss, params)

        print("Starting training...")
        for epoch in range(n_epochs):

            t = time.time()

            train_err = 0
            train_batches = 0

            for batch in iterate_minibatches(X_train, Y_train, batch_size, shuffle=True):
                if j % self.m == 0:
                    for mu in self.mu:
                        mu.set_value(0 * mu.get_value())

                    for mu_batch in iterate_minibatches(X_train, Y_train, batch_size, shuffle=False):
                        inputs, targets = mu_batch
                        train_mu(inputs, targets)
#                        train_mu(np.array(inputs.todense(), dtype=np.float32), np.array(targets, dtype=np.int32))
                    
                    for mu in self.mu:
                        mu.set_value(mu.get_value() / n)

                j += 1               
                inputs, targets = batch
                #print "learning_rate: ", 1. / self.L.get_value()
                
                current_loss = val_fn(inputs, targets)
#                current_loss = val_fn(np.array(inputs.todense(), dtype=np.float32), np.array(targets, dtype=np.int32))

                if self.adaptive: 
                    l_iter = 0
                    while True:

#                    print "learning_rate: ", 1. / self.L.get_value()

                        loss_next, sq_sum = L_fn(inputs, targets)
     #                   loss_next, sq_sum = L_fn(np.array(inputs.todense(), dtype=np.float32), np.array(targets, dtype=np.int32))
                        if loss_next <= current_loss - 0.5 * sq_sum / self.L.get_value():
                            break
                        else:
                            self.L.set_value(self.L.get_value() * 2)

                        l_iter += 1

                train_err += train_w(inputs, targets)
#                train_err += train_w(np.array(inputs.todense(), dtype=np.float32), np.array(targets, dtype=np.int32))
#                self.L.set_value(self.L.get_value() / 2)
                train_batches += 1
            
            val_err = 0
            val_batches = 0
            for i, batch in enumerate(iterate_minibatches(X_val, Y_val, batch_size, shuffle=True)):
                inputs, targets = batch
                val_err += val_fn(inputs, targets)
#                val_err += val_fn(np.array(inputs.todense(), dtype=np.float32), np.array(targets, dtype=np.int32))
                val_batches += 1

            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, n_epochs, time.time() - t))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

            train_error.append(train_err / train_batches)
            validation_error.append((val_err / val_batches, self.counted_gradient.get_value()))
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

            mu_updates[self.counted_gradient] = self.counted_gradient + 1

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
#            new_param = param - self.learning_rate * (grad - grad_tilde + mu)

            new_param = param - (1. / self.L) * (grad - grad_tilde + mu)
            w_updates[param] = new_param
            w_updates[param_tilde] = ifelse(T.eq(it % self.m, 0), new_param, param_tilde)
            
            w_updates[self.counted_gradient] = self.counted_gradient + 2
        
        if self.adaptive:
            w_updates[self.L] = self.L / 2

        self.it_num = it_num
        
        w_updates[it_num] = it
        return w_updates

    def make_L_fn(self, loss, params):
        grads = theano.grad(loss, params)

        params_next = [x - 1. / self.L * g for x, g in zip(params, grads)]
        loss_next = theano.clone(loss, replace=zip(params, params_next))
        sq_sum = sum((g**2).sum() for g in grads)

        return theano.function([self.input_var, self.target_var], [loss_next, sq_sum])
