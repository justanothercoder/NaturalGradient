import sys, os

import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as T 
import lasagne

from load_dataset import *
from neuralnet import train

from deep import DeepAutoEncoder
from sparse_autoencoder import SparseAutoEncoder
from neuralclassifier import NeuralClassifier

def main():
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')

    n_hidden = 500

    sparse_model = 'adam_sparse_3.0_not_denoising'
    classify_model = 'adam_classif_on_sparse'

    network = SparseAutoEncoder(n_input=X_train.shape[1], n_hidden=n_hidden, input_var=input_var)

    with np.load('models/model_%s.npz' % sparse_model) as f:
        param_values = [f['arr_%d' % j] for j in range(len(f.files))]
        lasagne.layers.set_all_param_values(network.output_layer, param_values)

    output_layer = lasagne.layers.DenseLayer(network.hidden_layer, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)
        
    prediction = lasagne.layers.get_output(output_layer)

    lambd = 0.0

    l2_reg = lasagne.regularization.regularize_layer_params(output_layer, lasagne.regularization.l2)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var) + lambd * l2_reg
    loss = loss.mean()

    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    params = params[2:]

    updates = lasagne.updates.adam(loss, params, learning_rate=0.01)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    if X_val is not None:
        test_prediction = lasagne.layers.get_output(output_layer, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
        test_loss = test_loss.mean()
        val_fn = theano.function([input_var, target_var], test_loss)
    else:
        val_fn = None
    
    n_epochs = 100
    batch_size = 500
    train_error, validation_error = train(
            X_train, y_train, X_val, y_val,
            train_fn, val_fn,
            n_epochs, batch_size=batch_size#, toprint=it
    )

    plt.plot(validation_error, label=classify_model)

    np.savez('models/model_%s.npz' % classify_model, *lasagne.layers.get_all_param_values(output_layer))
    np.savez('models/model_%s_val_error.npz' % classify_model, validation_error)

    accuracy = (lasagne.layers.get_output(output_layer, X_test, deterministic=True).eval().argmax(axis=1) == y_test).mean()
    print "Model: {}; accuracy: {}".format(classify_model, accuracy)

main()
