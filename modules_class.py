import tensorflow as tf
from tensorflow.contrib.distributions import Normal, kl
import numpy as np

def ph(shape, dtype=tf.float32, name=None):
    return tf.placeholder(dtype=dtype, shape=shape, name=name)


def rank(x):
    return len(x.get_shape())


def static_size(x, d):
    out = x.get_shape()[d].value
    assert out is not None, 'shape of dim d is not static'
    return out


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


class BatchNorm(Module):
    def __init__(self, beta_trainable=True, gamma_trainable=True, name='BatchNorm'):
        """
            n_out:       integer, depth of input maps
            input_rank:  rank of input x
            phase_train: boolean tf.Varialbe, true indicates training phase
            name:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        self.beta_trainable = beta_trainable
        self.gamma_trainable = gamma_trainable
        super(BatchNorm, self).__init__()

    def __call__(self, x, train_flag):
        with tf.name_scope(self.name) as scope:
            if self.called:
                assert rank(x) == self.input_rank
                assert static_size(x, -1) == self.input_size[-1]
            else:
                self.input_rank = rank(x)
                self.input_size = x.get_shape()
                self.output_size = x.get_shape()
                self.pool_axes = np.arange(self.input_rank-1).tolist()
                self.n_out = static_size(x, -1)
                self.beta = tf.Variable(tf.constant(0.0, shape=[self.n_out]),
                                            name='beta', trainable=self.beta_trainable)
                self.gamma = tf.Variable(tf.constant(1.0, shape=[self.n_out]),
                                            name='gamma', trainable=self.gamma_trainable)

            batch_mean, batch_var = tf.nn.moments(x, self.pool_axes, name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(train_flag,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, self.beta, self.gamma, 1e-3)

            normed.batch_mean = batch_mean
            normed.batch_var = batch_var
            normed.mean = mean
            normed.var = var
            super(BatchNorm, self).__call__()
            return normed


class Dropout(Module):
    def __init__(self, p_keep):
        self.p_keep = p_keep
        super(Dropout, self).__init__()

    def __call__(self, x, train_flag):
        mask = lambda: tf.to_float(tf.random_uniform(tf.shape(x)) < self.p_keep)
        # lambda: (x * mask) / self.p_keep,
        dropped = tf.cond(train_flag, 
                          lambda: (x * mask()) / self.p_keep,
                          lambda: x)
        super(Dropout, self).__call__()
        return dropped
