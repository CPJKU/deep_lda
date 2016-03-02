
import numpy as np


def batch_compute1(X, compute, batch_size):
    """ Batch compute data """

    # init results
    R = None

    # get number of samples
    n_samples = X.shape[0]

    # get input shape
    in_shape = list(X.shape)[1:]

    # get number of batches
    n_batches = int(np.ceil(float(n_samples) / batch_size))

    # iterate batches
    for i_batch in xrange(n_batches):

        # extract batch
        start_idx = i_batch * batch_size
        excerpt = slice(start_idx, start_idx + batch_size)
        E = X[excerpt]

        # append zeros if batch is to small
        n_missing = batch_size - E.shape[0]
        if n_missing > 0:
            E = np.vstack((E, np.zeros([n_missing] + in_shape, dtype=X.dtype)))

        # compute results on batch
        r = compute(E)

        # init result array
        if R is None:
            R = np.zeros([n_samples] + list(r.shape[1:]), dtype=r.dtype)

        # store results
        R[start_idx:start_idx+r.shape[0]] = r[0:batch_size-n_missing]

    return R


class BatchIterator(object):
    """
    Prototype for batch iterator
    """

    def __init__(self, batch_size, re_iterate=1, prepare=None):
        self.batch_size = batch_size

        if prepare is None:
            def prepare(x):
                return x
        self.prepare = prepare
        self.re_iterate = re_iterate

    def __call__(self, x, y=None):
        self.x, self.y = x, y
        self.n_batches = self.re_iterate * (self.x.shape[0] // self.batch_size)
        return self

    def __iter__(self):
        n_samples = self.x.shape[0]
        bs = self.batch_size

        for _ in xrange(self.re_iterate):
            for i in range((n_samples + bs - 1) / bs):
                sl = slice(i * bs, (i + 1) * bs)
                xb = self.x[sl]
                yb = self.y[sl] if self.y is not None else None

                if xb.shape[0] < self.batch_size:
                    n_missing = self.batch_size - xb.shape[0]

                    x_con  = self.x[0:n_missing]
                    xb = np.concatenate((xb, x_con))

                    if self.y is not None:
                        y_con = self.y[0:n_missing]
                        yb = np.concatenate((yb, y_con))

                yield self.transform(xb, yb)

    def transform(self, xb, yb):
        return self.prepare(xb), yb