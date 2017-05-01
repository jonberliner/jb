import numpy as np
rng = np.random.RandomState()
from datetime import datetime
import logging
import os
import pdb


class Bunch(dict):
    def __init__(self, *args, **kwds):
        super(Bunch, self).__init__(*args, **kwds)
        self.__dict__ = self


class Logger(object):
    def __init__(self, fname):
        # set up log file
        # LOG_NAME = 'inverted_mnist_v3_' + timestamp()
        import logging
        self._logger = logging
        logging.basicConfig(level=logging.DEBUG, filename=fname, filemode="a+",
                            format="%(asctime)-15s %(levelname)-8s %(message)s")
        logging.getLogger().addHandler(logging.StreamHandler())

    def log(self, string):
        self._logger.info(string)


def logger(fname):
    # set up log file
    # LOG_NAME = 'inverted_mnist_v3_' + timestamp()
    logging.basicConfig(level=logging.DEBUG, filename=fname, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())
    return logging

# http://stackoverflow.com/questions/11144513/numpy-cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
# (called cartesian_product2 there)
def cartesian1(arrays):
    """cartesian of arb amount of 1d arrays"""
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la])
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


coord_grid = lambda dim, res: cartesian1([np.linspace(-1,1,res)]*dim)


def quick_mnist(ch_dim=False, whiten=True):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    train_x = mnist.train.images
    test_x = mnist.test.images

    train_y = mnist.train.labels
    test_y = mnist.test.labels

    # whiten
    if whiten:
        mu = np.mean(train_x)
        sd = np.std(train_x)
        train_x = (train_x - mu) / sd
        test_x = (test_x - mu) / sd

    # add channel dim
    if ch_dim:
        train_x = np.expand_dims(train_x, 2)
        test_x = np.expand_dims(test_x, 2)

    return train_x, train_y, test_x, test_y


def prob_invert(x, p=0.5):
    assert x.max() <= 1., 'expects binary input'
    assert x.min() >= 0., 'expects binary input'
    invert = rng.rand(x.shape[0]) < p
    for ix, x0 in enumerate(x):
        x[ix] = x0<0.5 if invert[ix] else x0
    return x, invert.astype('float32')


def quick_inverted_mnist():
    # create train and test data
    train_x, train_y, test_x, test_y = quick_mnist(ch_dim=True, whiten=False)

    train_x, train_invert = prob_invert(train_x)
    test_x, test_invert = prob_invert(test_x)

timestamp = lambda: '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())

onehot = lambda n, ii: np.eye(n)[ii]


def cartesian(s0, s1, reverse_order=False):
    ns0 = s0.shape[0]
    ns1 = s1.shape[0]
    if reverse_order:
        rs0 = np.tile(s0, [ns1,1])
        rs1 = np.repeat(s1, ns0, axis=0)
    else:
        rs0 = np.repeat(s0, ns1, axis=0)
        rs1 = np.tile(s1, [ns0,1])

    out = np.concatenate([rs0, rs1], axis=1)
    return out


def unique_rows(a):
    """taken from 'http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array'"""
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    return a[idx]


class List(list):
    """
    from:
    http://code.activestate.com/recipes/579103-python-addset-attributes-to-list/
    A subclass of list that can accept additional attributes.
    Should be able to be used just like a regular list.
    The problem:
    a = [1, 2, 4, 8]
    a.x = "Hey!" # AttributeError: 'list' object has no attribute 'x'
    The solution:
    a = L(1, 2, 4, 8)
    a.x = "Hey!"
    print a       # [1, 2, 4, 8]
    print a.x     # "Hey!"
    print len(a)  # 4
    You can also do these:
    a = L( 1, 2, 4, 8 , x="Hey!" )                 # [1, 2, 4, 8]
    a = L( 1, 2, 4, 8 )( x="Hey!" )                # [1, 2, 4, 8]
    a = L( [1, 2, 4, 8] , x="Hey!" )               # [1, 2, 4, 8]
    a = L( {1, 2, 4, 8} , x="Hey!" )               # [1, 2, 4, 8]
    a = L( [2 ** b for b in range(4)] , x="Hey!" ) # [1, 2, 4, 8]
    a = L( (2 ** b for b in range(4)) , x="Hey!" ) # [1, 2, 4, 8]
    a = L( 2 ** b for b in range(4) )( x="Hey!" )  # [1, 2, 4, 8]
    a = L( 2 )                                     # [2]
    """
    def __new__(self, *args, **kwargs):
        return super(List, self).__new__(self, args, kwargs)

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            list.__init__(self, args[0])
        else:
            list.__init__(self, args)
        self.__dict__.update(kwargs)

    def __call__(self, **kwargs):
        self.__dict__.update(kwargs)
        return self


class Batcher(object):
    def __init__(self, X, batch_size, i_start=0, random_order=True):
        if type(X) == int:
            self.N = X
            self.X = np.arange(self.N)
        else:
            self.N = X.shape[0]
            self.X = X

        self.random_order = random_order
        self.order = np.arange(self.N)
        if self.random_order:
            rng.shuffle(self.order)
        self.batch_size = batch_size
        self.i_start = i_start
        self.get_i_end = lambda: min(self.i_start + self.batch_size, self.N)

        self.end_of_epoch = lambda: self.i_start == self.N
        self.batch_inds = None

    def __call__(self):
        inds = self.next_inds()
        return self.X[inds]

    def next_inds(self):
        i_end = self.get_i_end()
        if self.i_start == i_end:
            if self.random_order:
                rng.shuffle(self.order)
            self.i_start = 0
            i_end = self.get_i_end()
        batch_inds = self.order[self.i_start:i_end]
        batch_inds.sort()
        # increment
        self.i_start = i_end
        self.batch_inds = batch_inds
        return batch_inds


def clear_ticks(ax):
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


def mkdir_recursive(path):
    sub_path = os.path.dirname(path)
    if not os.path.exists(sub_path):
        mkdir_recursive(sub_path)
    if not os.path.exists(path):
        os.mkdir(path)
