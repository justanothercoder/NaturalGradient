import sys, os

import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as T 
import lasagne

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
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')

    network = build(input_var, n_hidden=300)

    methods = ['adam', 'momentum', 'nesterov_momentum', 'adagrad', 'rmsprop']

    n_images = 10
    
    for j in range(n_images):
        plt.subplot(len(methods) + 1, n_images, j + 1)
        #plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        if j == 0:
            plt.ylabel('original', rotation='horizontal')
        plt.imshow(X_train[j].reshape(28, 28), cmap='Greys')

    for i, model in enumerate(methods):
        with np.load('model_%s.npz' % model) as f:
            param_values = [f['arr_%d' % j] for j in range(len(f.files))]
            lasagne.layers.set_all_param_values(network, param_values)

        test_prediction = lasagne.layers.get_output(network, deterministic=True)

        for j in range(n_images):
            plt.subplot(len(methods) + 1, n_images, n_images * (i+1) + j + 1)
            #plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            if j == 0:
                plt.ylabel(model, rotation='horizontal')
            plt.imshow(lasagne.layers.get_output(network, X_train[j]).eval().reshape(28, 28), cmap='Greys')

#    n_images = 10
#    for i in range(n_images):
#        plt.subplot(n_images, 2, 2 * i + 1)
#        plt.axis('off')
#        plt.imshow(X_train[i].reshape(28, 28), cmap='Greys')
#        plt.subplot(n_images, 2, 2 * i + 2)
#        plt.axis('off')
#        plt.imshow(lasagne.layers.get_output(network, X_train[i]).eval().reshape(28, 28), cmap='Greys')
    
    plt.show()

main()
