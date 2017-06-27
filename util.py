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


def get_mnist(classes=None, n_per_class=None, n_per_test=None, onehot_label=True, seed=None, data_dir=''):
    classes = classes or np.arange(10).tolist()
    n_class = len(classes)
    for li in classes:
        assert li in np.arange(10)

    if n_per:
        if type(n_per) == list:
            assert len(n_per) == n_class
        else:
            n_per = [n_per]*n_class
    if n_per_test:
        if type(n_per_test) == list:
            assert len(n_per_test) == n_class
        else:
            n_per_test = [n_per_test] * n_class

    from tensorflow.examples.tutorials.mnist import input_data
    seed = seed or np.random.randint(1e8)
    rng = np.random.RandomState(seed)
    mnist = input_data.read_data_sets(os.path.join(data_dir, "MNIST_data/"), one_hot=False)
    dat = Bunch()
    dat.train = Bunch()
    dat.test = Bunch()
    dat.seed = seed


    dat.train.images = [None] * n_class
    dat.test.images = [None] * n_class
    dat.train.labels = [None] * n_class
    dat.test.labels = [None] * n_class
    for li, lab in enumerate(classes):
        i_train = np.where(mnist.train.labels == lab)[0]
        i_test = np.where(mnist.test.labels == lab)[0]
        dat.train.images[li] = mnist.train.images[i_train]
        dat.train.labels[li] = mnist.train.labels[i_train]
        dat.test.images[li] = mnist.test.images[i_test]
        dat.test.labels[li] = mnist.test.labels[i_test]

        if n_per:
            i_lab = rng.choice(dat.train.images[li].shape[0], n_per[li], replace=False)
            dat.train.images[li] = dat.train.images[li][i_lab]
            dat.train.labels[li] = dat.train.labels[li][i_lab]

        if n_per_test:
            i_lab = rng.choice(dat.test.images[li].shape[0], n_per_test[li], replace=False)
            dat.test.images[li] = dat.test.images[li][i_lab]
            dat.test.labels[li] = dat.test.labels[li][i_lab]

    dat.train.images = np.concatenate(dat.train.images, 0)
    dat.train.labels = np.concatenate(dat.train.labels, 0)
    dat.test.images = np.concatenate(dat.test.images, 0)
    dat.test.labels = np.concatenate(dat.test.labels, 0)
    dat.train.num_examples = dat.train.images.shape[0]
    dat.test.num_examples = dat.test.images.shape[0]

    if onehot_label:
        # TODO: should we leave as 10 classes or shrink?
        dat.train.labels = onehot(10, dat.train.labels)
        dat.test.labels = onehot(10, dat.test.labels)

    return dat



