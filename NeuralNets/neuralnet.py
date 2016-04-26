import sys, os, time
import numpy as np

import theano
import theano.tensor as T

import lasagne

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def build(input_var=None, n_hidden=200):
    input_layer = lasagne.layers.InputLayer(shape=(None, 784), input_var=input_var)
    hidden_layer = lasagne.layers.DenseLayer(input_layer, num_units=n_hidden, nonlinearity=lasagne.nonlinearities.sigmoid)
    output_layer = lasagne.layers.DenseLayer(hidden_layer, num_units=784, nonlinearity=lasagne.nonlinearities.sigmoid)
    return output_layer

def get_network_stats(X_train, X_val, X_test, n_epochs, n_hidden, objective, update, **update_params):
    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')

    network = build(input_var, n_hidden=n_hidden)

    prediction = lasagne.layers.get_output(network)
    #loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
    loss = objective(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    #updates = lasagne.updates.adam(loss, params, learning_rate=0.01)
    updates = update(loss, params, **update_params)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = objective(test_prediction, target_var)
    test_loss = test_loss.mean()

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], test_loss)

    train_error = []
    validation_error = []

    print("Starting training...")
    for epoch in range(n_epochs):

        train_err = 0
        train_batches = 0

        t = time.time()
        
        for batch in iterate_minibatches(X_train, X_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, X_val, 500, shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1

        train_error.append(train_err)
        validation_error.append(val_err)
        
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, n_epochs, time.time() - t))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, X_test, 500, shuffle=False):
        inputs, targets = batch
        err = val_fn(inputs, targets)
        test_err += err
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))

    return network, train_error, validation_error

# with np.load('model.npz') as f:
#   param_values = [f['arr_%d' % i] for i in range(len(f.files))]
# lasagne.layers.set_all_param_values(network, param_values)