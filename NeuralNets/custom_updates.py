import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

def custom_momentum(loss_or_grads, params, learning_rate=0.01, momentum=0.9):

    if not isinstance(loss_or_grads, list):
        grads = theano.grad(loss_or_grads, params)
    else:
        grad = loss_or_grads

#    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()    

    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad

    for param in params:
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

        x = momentum * velocity + updates[param]
        updates[velocity] = x - param
        updates[param] = x

    return updates

def custom_momentum_chlr(loss_or_grads, params, learning_rate=0.01, momentum=0.9, tau=0.0):

    if not isinstance(loss_or_grads, list):
        grads = theano.grad(loss_or_grads, params)
    else:
        grad = loss_or_grads

#    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()    

    it_num = theano.shared(1)

    for param, grad in zip(params, grads):
        updates[param] = param - T.cast(learning_rate / it_num**tau, 'float32') * grad

    for param in params:
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

        x = momentum * velocity + updates[param]
        updates[velocity] = x - param
        updates[param] = x

    updates[it_num] = it_num + 1

    return updates
