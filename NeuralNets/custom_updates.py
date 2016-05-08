import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from theano import pp

from collections import OrderedDict

def custom_momentum(loss_or_grads, params, learning_rate=0.01, momentum=0.9, tau=0.0):

    if not isinstance(loss_or_grads, list):
        grads = theano.grad(loss_or_grads, params)
    else:
        grads = loss_or_grads

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

def custom_nesterov_momentum(loss_or_grads, params, learning_rate=0.01, momentum=0.9, tau=0.0):
    
    if not isinstance(loss_or_grads, list):
        grads = theano.grad(loss_or_grads, params)
    else:
        grads = loss_or_grads

#    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()    

    it_num = theano.shared(1)

    for param, grad in zip(params, grads):
        updates[param] = param - T.cast(learning_rate / it_num**tau, 'float32') * grad

    for param in params:
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

        x = momentum * velocity + updates[param] - param
        updates[velocity] = x
        updates[param] = momentum * x + updates[param]

    updates[it_num] = it_num + 1

    return updates

def custom_adagrad(loss_or_grads, params, learning_rate=0.01, eps=1.0e-8):
    
    if not isinstance(loss_or_grads, list):
        grads = theano.grad(loss_or_grads, params)
    else:
        grads = loss_or_grads

    updates = OrderedDict()    

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        acc = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
        
        acc_new = acc + grad ** 2

        updates[acc] = acc_new
        updates[param] = param - learning_rate * grad / T.sqrt(acc_new + eps)

    return updates

def custom_rmsprop(loss_or_grads, params, learning_rate=0.01, rho=0.9, eps=1.0e-8):

    if not isinstance(loss_or_grads, list):
        grads = theano.grad(loss_or_grads, params)
    else:
        grad = loss_or_grads

    updates = OrderedDict()

    one = T.constant(1)

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        acc = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
        
        acc_new = rho * acc + (one - rho) * grad ** 2

        updates[acc] = acc_new
        updates[param] = param - learning_rate * grad / T.sqrt(acc_new + eps)

    return updates

def custom_adam(loss_or_grads, params, learning_rate=0.01, beta1=0.9, beta2=0.999, eps=1e-8):

    if not isinstance(loss_or_grads, list):
        grads = theano.grad(loss_or_grads, params)
    else:
        grads = loss_or_grads

    updates = OrderedDict()

    one = T.constant(1)

    it_num = theano.shared(np.cast['float32'](0.))
    it = it_num + 1
        
    alpha_t = learning_rate * T.sqrt(one - beta2**it) / (one - beta1**it)

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
    
        m_t = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
        v_t = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

        m_new = beta1 * m_t + (one - beta1) * grad
        v_new = beta2 * v_t + (one - beta2) * grad**2
        step = alpha_t * m_new / (T.sqrt(v_new) + eps)

        updates[m_t] = m_new
        updates[v_t] = v_new
        updates[param] = param - step
    
    updates[it_num] = it

    return updates

def custom_svrg2(loss, params, m, learning_rate=0.01, objective=None, data=None, target=None, getpred=None):

    pp(loss)
    
    grads = theano.grad(loss, params)
    n = data.shape[0]

    updates = OrderedDict()
    rng = RandomStreams(seed=149)

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)        
        mu = grad / n

        def oneStep(w):
            t = rng.choice(size=(1,), a=n)
            
            loss_part_tilde = objective(getpred(data[t], param), target[t])
            loss_part_tilde = loss_part_tilde.mean()
            g_tilde = theano.grad(loss_part_tilde, param)
        
            loss_part = objective(getpred(data[t], w), target[t])
            loss_part = loss_part.mean()
            g = theano.grad(loss_part, w)

            w = w - learning_rate * (g - g_tilde + mu)
            return w

        w_tilde, scan_updates = theano.scan(fn=oneStep, outputs_info=param, n_steps=m)

        updates.update(scan_updates)
        updates[param] = w_tilde[-1]

    return updates

def custom_svrg(loss, params, m=100, learning_rate=0.01):
    
    grads = theano.grad(loss, params)

    updates = OrderedDict()
    
    it_num = theano.shared(np.cast['int16'](0.))
    it = it_num + 1

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)

        mu = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

        grad_w_tilde = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
        new_grad_w_tilde = theano.ifelse.ifelse(T.eq(it, m), grad, grad_w_tilde)

        mu_acc = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)

        updates[param] = param - learning_rate * (grad - grad_w_tilde + mu)
        updates[grad_w_tilde] = new_grad_w_tilde

        updates[mu] = theano.ifelse.ifelse(T.eq(T.mod(it, m), 0), mu_acc, mu)
        updates[mu_acc] = theano.ifelse.ifelse(T.eq(T.mod(it, m), 0), 0*mu_acc, mu_acc + grad)

    updates[it_num] = theano.ifelse.ifelse(T.eq(it, m), np.cast['int16'](1), np.cast['int16'](m))

    return updates
