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
#    input_layer = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    input_layer = lasagne.layers.InputLayer(shape=(None, 784), input_var=input_var)
    hidden_layer = lasagne.layers.DenseLayer(input_layer, num_units=n_hidden, nonlinearity=lasagne.nonlinearities.sigmoid)
    output_layer = lasagne.layers.DenseLayer(hidden_layer, num_units=784, nonlinearity=lasagne.nonlinearities.sigmoid)
#    output_layer = lasagne.layers.DenseLayer(hidden_layer, num_units=784)

    return output_layer

def load_dataset():
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
#        data = data.reshape(-1, 1, 28, 28)
        data = data.reshape(-1, 784)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    
    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')

    network = build(input_var, n_hidden=300)

    prediction = lasagne.layers.get_output(network)
#    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
#    updates = lasagne.updates.momentum(loss, params, learning_rate=0.1, momentum=0.9)
    updates = lasagne.updates.adam(loss, params, learning_rate=0.01)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
#    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = lasagne.objectives.binary_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()

    #test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)    

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
#    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
#    val_fn = theano.function([input_var, target_var], [test_loss])
    val_fn = theano.function([input_var, target_var], test_loss)

    n_epochs = 100

    print("Starting training...")
    for epoch in range(n_epochs):

        train_err = 0
        train_batches = 0

        t = time.time()
        
#        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
        for batch in iterate_minibatches(X_train, X_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        val_err = 0
        val_acc = 0
        val_batches = 0
#        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
        for batch in iterate_minibatches(X_val, X_val, 500, shuffle=False):
            inputs, targets = batch
#            err, acc = val_fn(inputs, targets)
            err = val_fn(inputs, targets)
            val_err += err
#            val_acc += acc
            val_batches += 1
        
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, n_epochs, time.time() - t))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
#        print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
    
    test_err = 0
    test_acc = 0
    test_batches = 0
#    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
    for batch in iterate_minibatches(X_test, X_test, 500, shuffle=False):
        inputs, targets = batch
#        err, acc = val_fn(inputs, targets)
        err = val_fn(inputs, targets)
        test_err += err
#        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
#    print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))

    np.savez('model.npz', *lasagne.layers.get_all_param_values(network))

# with np.load('model.npz') as f:
#   param_values = [f['arr_%d' % i] for i in range(len(f.files))]
# lasagne.layers.set_all_param_values(network, param_values)

main()
