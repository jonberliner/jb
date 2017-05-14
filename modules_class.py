import tensorflow as tf
from tensorflow.contrib.distributions import Normal, kl
import numpy as np

class Module(object):
    def __init__(self, name=None):
        self.name = name
        self.input_size = None
        self.output_size = None
        self.called = False

    def __call__(self):
        self.called = True


class Linear(Module):
    def __init__(self, n_out, W_init='msra', name='linear'):
        super(Linear, self).__init__()
        self.n_out = n_out
        self.n_in = None
        self.output_size = (None, n_out)
        self.W_init = W_init

    def __call__(self, x):
        with tf.name_scope(self.name) as scope:
            if self.called:
                assert static_size(x, 1) == self.n_in
            else:
                self.input_size = x.get_shape()
                self.n_in = static_size(x, 1)
                if self.W_init == 'msra':
                    W0 = tf.random_normal([self.n_in, self.n_out]) * (tf.sqrt(2. / self.n_in))
                else:
                    raise ValueError('W_init must be in ["msra"]')
                self.W = tf.Variable(W0, name='W')
                self.b = tf.Variable(tf.zeros([self.n_out]), name='b')
            super(Linear, self).__call__()

            return tf.matmul(x, self.W) + self.b


class Sequential(object):
    def __init__(self, layers=None):
        self.layers = []
        if layers:
            [self.add(layer) for layer in layers]

    def __call__(self, x):
        hid = []
        out = x
        for layer in self.layers:
            out = layer(out)
            hid.append(out)
        out.hid = hid[:-1]
        return out

    def add(self, layer):
        self.layers.append(layer)


