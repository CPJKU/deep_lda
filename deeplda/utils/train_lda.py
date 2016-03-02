#!/usr/bin/env python
# train utils for DeepLDA

from __future__ import print_function

import os
import time
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt

import theano
import lasagne

# init color printer
class BColors:
    """
    Colored command line output formatting
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def __init__(self):
        """ Constructor """
        pass

    def print_colored(self, string, color):
        """ Change color of string """
        return color + string + BColors.ENDC

col = BColors()


def threaded_generator(generator, num_cached=10):
    """
    Threaded generator
    """
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    queue = Queue.Queue(maxsize=num_cached)
    end_marker = object()

    # define producer
    def producer():
        for item in generator:
            #item = np.array(item)  # if needed, create a copy here
            queue.put(item)
        queue.put(end_marker)

    # start producer
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer
    item = queue.get()
    while item is not end_marker:
        yield item
        queue.task_done()
        item = queue.get()


def generator_from_iterator(iterator):
    """
    Compile generator from iterator
    """
    for x in iterator:
        yield x


def threaded_generator_from_iterator(iterator, num_cached=10):
    """
    Compile threaded generator from iterator
    """
    generator = generator_from_iterator(iterator)
    return threaded_generator(generator, num_cached)


def accuracy_score(t, p):
    """
    Compute accuracy
    """
    return float(np.sum(p == t)) / len(p)


class LDA(object):
    """ LDA Class """

    def __init__(self, r=1e-3, n_components=None, verbose=False, show=False):
        """ Constructor """
        self.r = r
        self.n_components = n_components

        self.scalings_ = None
        self.coef_ = None
        self.intercept_ = None
        self.means = None

        self.verbose = verbose
        self.show = show

    def fit(self, X, y, X_te=None):
        """ Compute lda on hidden layer """

        # split into semi- and supervised- data
        X_all = X.copy()
        X = X[y >= 0]
        y = y[y >= 0]

        # get class labels
        classes = np.unique(y)

        # set number of components
        if self.n_components is None:
            self.n_components = len(classes) - 1

        # compute means
        means = []
        for group in classes:
            Xg = X[y == group, :]
            means.append(Xg.mean(0))
        self.means = np.asarray(means)

        # compute covs
        covs = []
        for group in classes:
            Xg = X[y == group, :]
            Xg = Xg - np.mean(Xg, axis=0)
            covs.append(np.cov(Xg.T))

        # within scatter
        Sw = np.average(covs, axis=0)

        # total scatter
        X_all = X_all - np.mean(X_all, axis=0)
        if X_te is not None:
            St = np.cov(np.concatenate((X_all, X_te)).T)
        else:
            St = np.cov(X_all.T)

        # between scatter
        Sb = St - Sw

        # cope for numerical instability
        Sw += np.identity(Sw.shape[0]) * self.r

        # compute eigen decomposition
        from scipy.linalg.decomp import eigh
        evals, evecs = eigh(Sb, Sw)

        # sort eigen vectors according to eigen values
        evecs = evecs[:, np.argsort(evals)[::-1]]

        # normalize eigen vectors
        evecs /= np.apply_along_axis(np.linalg.norm, 0, evecs)

        # compute lda data
        self.scalings_ = evecs
        self.coef_ = np.dot(self.means, evecs).dot(evecs.T)
        self.intercept_ = (-0.5 * np.diag(np.dot(self.means, self.coef_.T)))

        if self.verbose:
            top_k_evals = evals[-self.n_components:]
            print("LDA-Eigenvalues (Train):", np.array_str(top_k_evals, precision=2, suppress_small=True))
            print("Ratio min(eigval)/max(eigval): %.3f, Mean(eigvals): %.3f" % (top_k_evals.min() / top_k_evals.max(), top_k_evals.mean()))

        if self.show:
            plt.figure("Eigenvalues")
            ax = plt.subplot(111)
            top_k_evals /= np.sum(top_k_evals)
            plt.plot(range(self.n_components), top_k_evals, 'bo-')
            plt.grid('on')
            plt.xlabel('Eigenvalue', fontsize=20)
            plt.ylabel('Explained Discriminative Variance', fontsize=20)
            plt.ylim([0.0, 1.05 * np.max(top_k_evals)])

            ax.tick_params(axis='x', labelsize=18)
            ax.tick_params(axis='y', labelsize=18)

        return evals

    def transform(self, X):
        """ transform data """
        X_new = np.dot(X, self.scalings_)
        return X_new[:, :self.n_components]

    def predict_proba(self, X):
        """ estimate probability """
        prob = -(np.dot(X, self.coef_.T) + self.intercept_)
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
        return prob

    def predict_log_proba(self, X):
        """ estimate log probability """
        return np.log(self.predict_proba(X))


def create_iter_functions(l_out, l_in, y_tensor_type, objective, learning_rate, l_2, compute_updates):
    """ Create functions for training, validation and testing to iterate one epoch. """

    # init target tensor
    targets = y_tensor_type('y')

    # compute train costs
    tr_output = lasagne.layers.get_output(l_out, deterministic=False)
    tr_cost = objective(tr_output, targets)

    # compute validation costs
    va_output = lasagne.layers.get_output(l_out, deterministic=True)
    va_cost = objective(va_output, targets)

    # collect all parameters of net and compute updates
    all_params = lasagne.layers.get_all_params(l_out, trainable=True)

    # add weight decay
    if l_2 is not None:
        tr_cost += l_2 * lasagne.regularization.apply_penalty(all_params, lasagne.regularization.l2)

    # compute updates from gradients
    all_grads = lasagne.updates.get_or_compute_grads(tr_cost, all_params)
    updates = compute_updates(all_grads, all_params, learning_rate)

    # compile iter functions
    tr_outputs = [tr_cost]
    iter_train = theano.function([l_in.input_var, targets], tr_outputs, updates=updates)

    va_outputs = [va_cost, va_output]
    iter_valid = theano.function([l_in.input_var, targets], va_outputs)

    # compile output function
    compute_output = theano.function([l_in.input_var], va_output)

    return dict(train=iter_train, valid=iter_valid, test=iter_valid, compute_output=compute_output)


def train(iter_funcs, dataset, train_batch_iter, valid_batch_iter, r):
    """
    Train the model with `dataset` with mini-batch training.
    Each mini-batch has `batch_size` recordings.
    """
    import sys
    import time

    for epoch in itertools.count(1):

        # iterate train batches
        batch_train_losses = []
        iterator = train_batch_iter(dataset['X_train'], dataset['y_train'])
        generator = threaded_generator_from_iterator(iterator)

        start, after = time.time(), time.time()
        for i_batch, (X_b, y_b) in enumerate(generator):
            batch_res = iter_funcs['train'](X_b, y_b)
            batch_train_losses.append(batch_res[0])
            after = time.time()
            train_time = (after-start)

            # report loss during training
            perc = 100 * (float(i_batch) / train_batch_iter.n_batches)
            dec = int(perc // 4)
            progbar = "|" + dec * "#" + (25-dec) * "-" + "|"
            vals = (perc, progbar, train_time, np.mean(batch_train_losses))
            loss_str = " (%d%%) %s time: %.2fs, loss: %.5f" % vals
            print(col.print_colored(loss_str, col.WARNING), end="\r")
            sys.stdout.flush()

        print("\x1b[K", end="\r")
        avg_train_loss = np.mean(batch_train_losses)

        # lda evaluation (accuracy based)

        # iterate validation batches
        batch_valid_losses = []
        iterator = valid_batch_iter(dataset['X_valid'], dataset['y_valid'])
        generator = threaded_generator_from_iterator(iterator)
        net_output_va, y_va = None, np.zeros(0, dtype=np.int32)
        for X_b, y_b in generator:
            batch_res = iter_funcs['valid'](X_b, y_b)
            batch_valid_losses.append(batch_res[0])

            y_va = np.concatenate((y_va, y_b))
            net_output = iter_funcs['compute_output'](X_b)
            if net_output_va is None:
                net_output_va = net_output
            else:
                net_output_va = np.vstack((net_output_va, net_output))

        avg_valid_loss = np.mean(batch_valid_losses)

        # compute train set net output
        iterator = train_batch_iter(dataset['X_train'], dataset['y_train'])
        generator = threaded_generator_from_iterator(iterator)
        net_output_tr, y_tr = None, np.zeros(0, dtype=np.int32)
        for i_batch, (X_b, y_b) in enumerate(generator):
            y_tr = np.concatenate((y_tr, y_b))
            net_output = iter_funcs['compute_output'](X_b)
            if net_output_tr is None:
                net_output_tr = net_output
            else:
                net_output_tr = np.vstack((net_output_tr, net_output))

        # fit lda on net output
        print("")
        dlda = LDA(r=r, n_components=None, verbose=True)
        evals = dlda.fit(net_output_tr, y_tr)

        # predict on train set
        proba = dlda.predict_proba(net_output_tr[y_tr >= 0])
        y_tr_pr = np.argmax(proba, axis=1)
        tr_acc = 100 * accuracy_score(y_tr[y_tr >= 0], y_tr_pr)

        # predict on validation set
        proba = dlda.predict_proba(net_output_va)
        y_va_pr = np.argmax(proba, axis=1)
        va_acc = 100 * accuracy_score(y_va, y_va_pr)

        # estimate overfitting
        overfit = va_acc / tr_acc

        # collect results
        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'train_acc': tr_acc,
            'valid_loss': avg_valid_loss,
            'valid_acc': va_acc,
            'overfitting': overfit,
            'eigenvalues': evals
        }


def fit(l_out, l_in, data, objective, y_tensor_type,
        train_batch_iter, valid_batch_iter,
        r=1e-3, num_epochs=100, patience=20,
        learn_rate=0.01, update_learning_rate=None,
        l_2=None, compute_updates=None,
        exp_name='ff', out_path=None, dump_file=None):
    """ Train model """

    # log model evolution
    log_file = os.path.join(out_path, 'results.pkl')

    print("\n")
    print(col.print_colored("Running Test Case: " + exp_name, BColors.UNDERLINE))

    # adaptive learning rate
    learning_rate = theano.shared(np.float32(learn_rate))
    if update_learning_rate is None:
        def update_learning_rate(lr, e):
            return lr
    learning_rate.set_value(update_learning_rate(learn_rate, 0))

    # initialize evaluation output
    pred_tr_err, pred_val_err, overfitting = [], [], []
    tr_accs, va_accs = [], []
    eigenvalues = []

    print("Building model and compiling functions...")
    iter_funcs = create_iter_functions(l_out, l_in, y_tensor_type, objective, learning_rate=learning_rate,
                                       l_2=l_2, compute_updates=compute_updates)

    print("Starting training...")
    now = time.time()
    try:

        # initialize early stopping
        last_improvement = 0
        best_model = lasagne.layers.get_all_param_values(l_out)

        # iterate training epochs
        prev_acc_tr, prev_acc_va = 0.0, 0.0
        for epoch in train(iter_funcs, data, train_batch_iter, valid_batch_iter, r):

            print("Epoch {} of {} took {:.3f}s".format(
                epoch['number'], num_epochs, time.time() - now))
            now = time.time()

            # update learning rate
            learn_rate = update_learning_rate(learn_rate, epoch['number'])
            learning_rate.set_value(learn_rate)

            # --- collect train output ---

            tr_loss, va_loss = epoch['train_loss'], epoch['valid_loss']
            train_acc, valid_acc = epoch['train_acc'], epoch['valid_acc']
            overfit = epoch['overfitting']

            # prepare early stopping
            if valid_acc >= prev_acc_va:
                last_improvement = 0
                best_model = lasagne.layers.get_all_param_values(l_out)

                # dump net parameters during training
                if dump_file is not None:
                    with open(dump_file, 'w') as fp:
                        params = lasagne.layers.get_all_param_values(l_out)
                        pickle.dump(params, fp)

            # increase improvement counter
            last_improvement += 1

            # plot train output
            if train_acc is None:
                txt_tr = 'costs_tr %.5f' % tr_loss
            else:
                txt_tr = 'costs_tr %.5f (%.3f), ' % (tr_loss, train_acc)
            if train_acc >= prev_acc_tr:
                txt_tr = col.print_colored(txt_tr, BColors.OKGREEN)
                prev_acc_tr = train_acc

            if valid_acc is None:
                txt_val = ''
            else:
                txt_val = 'costs_val %.5f (%.3f), tr/val %.3f' % (va_loss, valid_acc, overfit)
            if valid_acc >= prev_acc_va:
                txt_val = col.print_colored(txt_val, BColors.OKGREEN)
                prev_acc_va = valid_acc

            print('  lr: %.5f' % learn_rate)
            print('  ' + txt_tr + txt_val)

            # collect model evolution data
            tr_accs.append(train_acc)
            va_accs.append(valid_acc)
            pred_tr_err.append(tr_loss)
            pred_val_err.append(va_loss)
            overfitting.append(overfit)
            eigenvalues.append(epoch['eigenvalues'])

            # --- early stopping: preserve best model ---
            if last_improvement > patience:
                print(col.print_colored("Early Stopping!", BColors.WARNING))
                status = "Epoch: %d, Best Validation Accuracy: %.3f" % (epoch['number'], prev_acc_va)
                print(col.print_colored(status, BColors.WARNING))
                break

            # maximum number of epochs reached
            if epoch['number'] >= num_epochs:
                break

            # shuffle train data
            if not hasattr(data['X_train'], 'reset_batch_generator'):
                rand_idx = np.random.permutation(data['X_train'].shape[0])
                data['X_train'] = data['X_train'][rand_idx]
                data['y_train'] = data['y_train'][rand_idx]

            # save results
            exp_res = dict()
            exp_res['pred_tr_err'] = pred_tr_err
            exp_res['tr_accs'] = tr_accs
            exp_res['pred_val_err'] = pred_val_err
            exp_res['va_accs'] = va_accs
            exp_res['overfitting'] = overfitting
            exp_res['eigenvalues'] = eigenvalues

            with open(log_file, 'w') as fp:
                pickle.dump(exp_res, fp)

    except KeyboardInterrupt:
        pass

    # set net to best weights
    lasagne.layers.set_all_param_values(l_out, best_model)

    # evaluate on test set
    test_losses, test_acc = [], []
    iterator = valid_batch_iter(data['X_test'], data['y_test'])
    for X_b, y_b in iterator:
        loss_te = iter_funcs['test'](X_b, y_b)
        test_losses.append(loss_te[0])
        if len(loss_te) > 1:
            test_acc.append(loss_te[1])

    # compute evaluation measures
    avg_loss_te = np.mean(test_losses)
    avg_acc_te = np.mean(test_acc)

    print("--------------------------------------------")
    print('Loss on Test-Set: %.5f (%.3f)' % (avg_loss_te, avg_acc_te))
    print("--------------------------------------------\n")

    if out_path is not None:

        # add test results and save results
        exp_res['avg_loss_te'] = avg_loss_te
        exp_res['avg_acc_te'] = avg_acc_te

        with open(log_file, 'w') as fp:
            pickle.dump(exp_res, fp)

    return l_out, prev_acc_va
