import sys, os, time
import numpy as np

import theano
import theano.tensor as T

import lasagne

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    #assert len(inputs) == len(targets)
    assert inputs.shape[0] == len(targets)
    if shuffle:
#        indices = np.arange(len(inputs))
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
#    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def train(X_train, Y_train, X_val, Y_val, train_fn, val_fn, n_epochs, batch_size=500, verbose=True, toprint=None):
    train_error = []
    validation_error = []

    gradient_times = 0

    if verbose:
        print("Starting training...")
    for epoch in range(n_epochs):

        train_err = 0
        train_batches = 0

        t = time.time()
        
        for batch in iterate_minibatches(X_train, Y_train, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
#            train_err += train_fn(np.array(inputs.todense(), dtype=np.float32), np.array(targets, dtype=np.int32))
            gradient_times += 1
            train_batches += 1

            if toprint is not None:
                print toprint.get_value()

        train_error.append(train_err / train_batches)

        if X_val is not None:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_val, Y_val, batch_size, shuffle=False):
                inputs, targets = batch
                err = val_fn(inputs, targets)
#                err = val_fn(np.array(inputs.todense(), dtype=np.float32), np.array(targets, dtype=np.int32))

#                err, acc = val_fn(inputs, targets)
                val_err += err
#                val_acc += acc
                val_batches += 1
       
            validation_error.append((val_err / val_batches, gradient_times))

        if verbose:
            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, n_epochs, time.time() - t))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            if X_val is not None:
                print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
                print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

    return train_error, validation_error
