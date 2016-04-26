#!/usr/bin/env python

import numpy as np

import theano
import theano.tensor as T
from theano.tensor import slinalg

import lasagne
try:
    from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
except:
    from lasagne.layers.conv import Conv2DLayer as Conv2DLayer
from lasagne.layers.normalization import batch_norm

from batch_iterators import BatchIterator

EXP_NAME = 'stl10_dlda'
INI_LEARNING_RATE = 0.1
BATCH_SIZE = 200
MOMENTUM = 0.9
MAX_EPOCHS = 1000
PATIENCE = MAX_EPOCHS
X_TENSOR_TYPE = T.tensor4
Y_TENSOR_TYPE = T.ivector
INPUT_SHAPE = [3, 96, 96]

L2 = 0.0001

n_classes = 10
n_components = n_classes - 1
r = 1e-3

init_conv = lasagne.init.HeNormal


def build_model(batch_size=BATCH_SIZE):
    """ Compile net architecture """

    # --- conv layers ---
    l_in = lasagne.layers.InputLayer(shape=(batch_size, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]), name='Input')

    # --- conv layers ---
    net = Conv2DLayer(l_in, num_filters=64, filter_size=3, pad=1, W=init_conv(gain='relu'),
                      nonlinearity=lasagne.nonlinearities.rectify, name='Conv')
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=64, filter_size=3, pad=1, W=init_conv(gain='relu'),
                      nonlinearity=lasagne.nonlinearities.rectify, name='Conv')
    net = batch_norm(net)
    net = lasagne.layers.MaxPool2DLayer(net, pool_size=2, name='Pool')
    net = lasagne.layers.DropoutLayer(net, p=0.25, name='Dropout')

    net = Conv2DLayer(net, num_filters=128, filter_size=3, pad=1, W=init_conv(gain='relu'),
                      nonlinearity=lasagne.nonlinearities.rectify, name='Conv')
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=128, filter_size=3, pad=1, W=init_conv(gain='relu'),
                      nonlinearity=lasagne.nonlinearities.rectify, name='Conv')
    net = batch_norm(net)
    net = lasagne.layers.MaxPool2DLayer(net, pool_size=2, name='Pool')
    net = lasagne.layers.DropoutLayer(net, p=0.25, name='Dropout')

    net = Conv2DLayer(net, num_filters=256, filter_size=3, pad=1, W=init_conv(gain='relu'),
                      nonlinearity=lasagne.nonlinearities.rectify, name='Conv')
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=256, filter_size=3, pad=1, W=init_conv(gain='relu'),
                      nonlinearity=lasagne.nonlinearities.rectify, name='Conv')
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=256, filter_size=3, pad=1, W=init_conv(gain='relu'),
                      nonlinearity=lasagne.nonlinearities.rectify, name='Conv')
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=256, filter_size=3, pad=1, W=init_conv(gain='relu'),
                      nonlinearity=lasagne.nonlinearities.rectify, name='Conv')
    net = batch_norm(net)
    net = lasagne.layers.MaxPool2DLayer(net, pool_size=2, name='Pool')
    net = lasagne.layers.DropoutLayer(net, p=0.25, name='Dropout')

    net = Conv2DLayer(net, num_filters=1024, filter_size=3, pad=0, W=init_conv(gain='relu'),
                      nonlinearity=lasagne.nonlinearities.rectify, name='Conv')
    net = batch_norm(net)
    net = lasagne.layers.DropoutLayer(net, p=0.5, name='Dropout')
    net = Conv2DLayer(net, num_filters=1024, filter_size=1, pad=0, W=init_conv(gain='relu'),
                      nonlinearity=lasagne.nonlinearities.rectify, name='Conv')
    net = batch_norm(net)
    net = lasagne.layers.DropoutLayer(net, p=0.5, name='Dropout')

    # --- classification layer ---
    net = Conv2DLayer(net, num_filters=10, filter_size=1, W=init_conv(gain='relu'),
                      nonlinearity=lasagne.nonlinearities.rectify, name='Conv')
    net = batch_norm(net)
    net = lasagne.layers.Pool2DLayer(net, pool_size=10, ignore_border=False,
                                     mode='average_exc_pad', name='GlobalAveragePool')
    net = lasagne.layers.FlattenLayer(net, name='Flatten')

    return net, l_in


def objective(Xt, yt):
    """
    DeepLDA optimization target
    """

    # init groups
    groups = T.arange(0, n_classes)

    def compute_cov(group, Xt, yt):
        """
        Compute class covariance matrix for group
        """
        Xgt = Xt[T.eq(yt, group).nonzero()]
        Xgt_bar = Xgt - T.mean(Xgt, axis=0)
        m = T.cast(Xgt_bar.shape[0], 'float32')
        return (1.0 / (m - 1)) * T.dot(Xgt_bar.T, Xgt_bar)

    # scan over groups
    covs_t, updates = theano.scan(fn=compute_cov, outputs_info=None,
                                  sequences=[groups], non_sequences=[Xt, yt])

    # compute average covariance matrix (within scatter)
    Sw_t = T.mean(covs_t, axis=0)

    # compute total scatter
    Xt_bar = Xt - T.mean(Xt, axis=0)
    m = T.cast(Xt_bar.shape[0], 'float32')
    St_t = (1.0 / (m - 1)) * T.dot(Xt_bar.T, Xt_bar)

    # compute between scatter
    Sb_t = St_t - Sw_t

    # cope for numerical instability (regularize)
    Sw_t += T.identity_like(Sw_t) * r

    # compute eigenvalues
    evals_t = slinalg.eigvalsh(Sb_t, Sw_t)

    # get eigenvalues
    top_k_evals = evals_t[-n_components:]

    # maximize variance between classes
    # (k smallest eigenvalues below threshold)
    thresh = T.min(top_k_evals) + 1.0
    top_k_evals = top_k_evals[(top_k_evals <= thresh).nonzero()]
    costs = -T.mean(top_k_evals)

    return costs


def update_learning_rate(lr, epoch=None):
    """ Update learning rate """
    if epoch > 0 and np.mod(epoch, 100) == 0:
        return lr / 2
    else:
        return lr


def compute_updates(all_grads, all_params, learning_rate):
    """ Compute updates from gradients """
    return lasagne.updates.nesterov_momentum(all_grads, all_params, learning_rate, momentum=0.9)


def valid_batch_iterator(batch_size=BATCH_SIZE):
    """ Compile batch iterator """
    return BatchIterator(batch_size=batch_size, prepare=None, re_iterate=1)


def train_batch_iterator(batch_size=BATCH_SIZE):
    """ Compile batch iterator """

    def prepare(x):
        """ randomly flip images """
        fl = np.random.randint(0, 2, x.shape[0])
        for i in xrange(x.shape[0]):
            if fl[i] == 1:
                x[i] = x[i, :, :, ::-1]
        return x

    return BatchIterator(batch_size=batch_size, prepare=prepare, re_iterate=1)

