#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import numpy as np


def download_file(filename, source):
    """
    Load file from url
    """
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    print("Downloading %s ... " % filename, end='')
    urlretrieve(source + filename, filename)
    print("done!")


def load_mnist(k60=False):
    """
    Load mnist
    """
    import gzip
    import pickle

    # download mnist dataset if not available
    if not os.path.exists('mnist.pkl.gz'):
        download_file(filename='mnist.pkl.gz',
                      source='http://deeplearning.net/data/mnist/')

    # load data
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)

    # load labels
    y_train = np.asarray(train_set[1], dtype=np.int32)
    y_valid = np.asarray(valid_set[1], dtype=np.int32)
    y_test = np.asarray(test_set[1], dtype=np.int32)

    # load images
    X_tr, X_va, X_te = train_set[0].astype(np.float32), valid_set[0].astype(np.float32), test_set[0].astype(np.float32)

    # reshape to conv-net format
    X_tr = X_tr.reshape((-1, 1, 28, 28))
    X_va = X_va.reshape((-1, 1, 28, 28))
    X_te = X_te.reshape((-1, 1, 28, 28))

    # use 60k images for training
    if k60:
        X_tr = np.concatenate((X_tr, X_va))
        y_train = np.concatenate((y_train, y_valid))
        X_va = X_te
        y_valid = y_test

    print(" #Train Samples:", X_tr.shape)
    print(" #Valid Samples:", X_va.shape)
    print(" #Test Samples: ", X_te.shape)

    return dict(X_train=X_tr, y_train=y_train,
                X_valid=X_va, y_valid=y_valid,
                X_test=X_te, y_test=y_test)


def load_cifar10(k50=False):
    """
    Load cifar10
    """
    import pickle
    import tarfile
    data_root = 'cifar-10-batches-py'
    fold = 5

    def reshape_cifar(X):
        """ Reshape images """
        X = X.reshape(-1, 3072)
        X = X.reshape((X.shape[0], 32, 32, 3), order='F')
        return np.transpose(X, axes=(0, 2, 1, 3))

    # download mnist dataset if not available
    if not os.path.exists('cifar-10-python.tar.gz'):
        download_file(filename='cifar-10-python.tar.gz',
                      source='https://www.cs.toronto.edu/~kriz/')

        print("Extracting files ... ", end='')
        tar = tarfile.open('cifar-10-python.tar.gz')
        tar.extractall()
        tar.close()
        print("done!")

    # load all training batches
    X_train = np.zeros((0, 3072), dtype=np.uint8)
    y_train = []
    for batch in xrange(1, 6):
        file_path = os.path.join(data_root, 'data_batch_' + str(batch))

        with open(file_path, 'rb') as fo:
            batch_data = pickle.load(fo)
            X_train = np.vstack((X_train, batch_data['data']))
            y_train += batch_data['labels']

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)

    # load test batch
    file_path = os.path.join(data_root, 'test_batch')
    with open(file_path, 'rb') as fo:
            batch_data = pickle.load(fo)
            X_test = batch_data['data'].astype(np.float32)
            X_test = X_test.reshape(-1, 3072)
            y_test = batch_data['labels']
            y_test = np.asarray(y_test, dtype=np.int32)

    # normalize data do range (0,1)
    X_train /= 255.0
    X_test /= 255.0

    # compile validation data
    if not k50:
        va_idxs = range(fold - 1, fold - 1 + 10000)
        tr_idxs = np.setdiff1d(range(50000), va_idxs)
        X_valid = X_train[va_idxs]
        y_valid = y_train[va_idxs]
        X_train = X_train[tr_idxs]
        y_train = y_train[tr_idxs]
    else:
        X_valid = X_test
        y_valid = y_test

    # reshape datasets to rgb images
    X_train = reshape_cifar(X_train)
    X_valid = reshape_cifar(X_valid)
    X_test = reshape_cifar(X_test)

    X_train = np.transpose(np.swapaxes(X_train, 1, 3), (0, 1, 3, 2))
    X_valid = np.transpose(np.swapaxes(X_valid, 1, 3), (0, 1, 3, 2))
    X_test = np.transpose(np.swapaxes(X_test, 1, 3), (0, 1, 3, 2))

    # compile train data
    train_set = (X_train, y_train)
    valid_set = (X_valid, y_valid)
    test_set = (X_test, y_test)

    # extract data
    X_tr, y_tr = train_set[0], train_set[1]
    X_va, y_va = valid_set[0], valid_set[1]
    X_te, y_te = test_set[0], test_set[1]

    print(" #Train Samples:", X_tr.shape)
    print(" #Valid Samples:", X_va.shape)
    print(" #Test Samples: ", X_te.shape)

    return dict(X_train=X_tr, y_train=y_tr,
                X_valid=X_va, y_valid=y_va,
                X_test=X_te, y_test=y_te)


def load_stl10(fold=0, normalize=True):
    """
    Load stl10
    """
    import tarfile
    from scipy.io import loadmat
    data_root = 'stl10_matlab'

    # download dataset
    if not os.path.exists('stl10_matlab.tar.gz'):
        download_file(filename='stl10_matlab.tar.gz',
                      source='http://ai.stanford.edu/~acoates/stl10/')

        print("Extracting files ... ", end='')
        tar = tarfile.open('stl10_matlab.tar.gz')
        tar.extractall()
        tar.close()
        print("done!")

    # load train data
    file_path = os.path.join(data_root, 'train.mat')
    train_data = loadmat(file_path)
    X_train_raw = train_data['X'].astype(np.float32)
    y_train_raw = train_data['y'].astype(np.int32)
    fold_idxs = train_data['fold_indices']

    # compile validation data
    indices = fold_idxs[0, fold].flatten() - 1
    X_train = X_train_raw[indices]
    y_train = y_train_raw[indices].flatten()

    va_indices = np.setdiff1d(range(5000), indices)
    X_valid = X_train_raw[va_indices]
    y_valid = y_train_raw[va_indices].flatten()

    # load test data
    file_path = os.path.join(data_root, 'test.mat')
    test_data = loadmat(file_path)
    X_test    = test_data['X'].astype(np.float32)
    y_test    = test_data['y'].astype(np.int32).flatten()

    # normalize data to range (0,1)
    if normalize:
        X_train /= 255
        X_valid /= 255
        X_test /= 255

    # reshape datasets to rgb images
    X_train = X_train.reshape((X_train.shape[0], 96, 96, 3), order='F')
    X_test  = X_test.reshape((X_test.shape[0], 96, 96, 3), order='F')
    X_valid = X_valid.reshape((X_valid.shape[0], 96, 96, 3), order='F')

    # convert to conv-net format
    X_train = np.transpose(np.swapaxes(X_train, 1, 3), (0, 1, 3, 2))
    X_valid = np.transpose(np.swapaxes(X_valid, 1, 3), (0, 1, 3, 2))
    X_test = np.transpose(np.swapaxes(X_test, 1, 3), (0, 1, 3, 2))

    # shift labels to start with 0
    y_train -= 1
    y_valid -= 1
    y_test -= 1

    # compile train data
    train_set = (X_train, y_train)
    valid_set = (X_valid, y_valid)
    test_set  = (X_test, y_test)

    # extract data
    X_tr, y_tr = train_set[0], train_set[1]
    X_va, y_va = valid_set[0], valid_set[1]
    X_te, y_te = test_set[0], test_set[1]

    print(" #Train Samples:", X_tr.shape)
    print(" #Valid Samples:", X_va.shape)
    print(" #Test Samples: ", X_te.shape)

    return dict(X_train=X_tr, y_train=y_tr,
                X_valid=X_va, y_valid=y_va,
                X_test=X_te, y_test=y_te)


if __name__ == '__main__':
    """ main """
    data = load_cifar10()
