import tensorflow as tf
from tensorflow.contrib.distributions import Normal, kl
import numpy as np

# # TODO: can we set up these flags?
# train_flag = tf.Variable(True, trainable=False)
# set_train_flag = lambda sess, flag: sess.run(tf.assign(TRAIN_FLAG, flag))

train_flag = tf.placeholder(tf.bool)

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
        self.input_shape = None
        self.output_shape = None
        self.called = False

    def __call__(self):
        self.called = True


class SameShape(Module):
    """a module where what comes out is same shape as what went in"""
    def __call__(self, x):
        self.input_shape = x.get_shape()
        self.output_shape = x.get_shape()
        super(SameShape, self).__call__()


class Linear(Module):
    def __init__(self, n_out, W_init='msra', name='linear'):
        super(Linear, self).__init__()
        self.n_out = n_out
        self.n_in = None
        self.output_shape = (None, n_out)
        self.W_init = W_init

    def __call__(self, x):
        with tf.name_scope(self.name) as scope:
            if self.called:
                assert static_size(x, 1) == self.n_in
            else:
                self.input_shape = x.get_shape()
                self.n_in = static_size(x, 1)
                if self.W_init == 'msra':
                    W0 = tf.random_normal([self.n_in, self.n_out]) * (tf.sqrt(2. / self.n_in))
                else:
                    raise ValueError('W_init must be in ["msra"]')
                self.W = tf.Variable(W0, name='W')
                self.b = tf.Variable(tf.zeros([self.n_out]), name='b')
            super(Linear, self).__call__()

            return tf.matmul(x, self.W) + self.b


class FCLayer(Module):
    # TODO: decide if we want to break into containers, stacks, etc
    def __init__(self, n_out, act_fn=tf.nn.relu, bn=False, p_drop=False, name='FCLayer'):
        with tf.name_scope(name) as scope:
            super(FCLayer, self).__init__(name)
            self.n_out = n_out
            self.bn = bn
            self.p_drop = p_drop
            self.drop = p_drop is not None

            self.linear = Linear(n_out)
            self.batch_norm = BatchNorm() if self.bn else None
            self.act_fn = act_fn
            self.dropout = Dropout(p_drop) if self.drop else None


    def __call__(self, x, train_flag=train_flag):
        # if self.bn or self.drop:
        #     assert train_flag is not None, 'must provide train_flag if using batchnorm or dropout'
        if self.called:
            assert rank(x) == self.input_rank
            assert static_size(x, 1) == self.n_in
        else:
            self.input_rank = rank(x)
            self.input_shape = x.get_shape()
            self.n_in = static_size(x, 1)

        preact = self.linear(x)
        bn_preact = self.batch_norm(preact) if self.bn else None
        act = self.act_fn(bn_preact) if self.bn else self.act_fn(preact)
        dropped = self.dropout(act) if self.drop else None

        out = dropped if self.drop else act
        out.preact = preact
        out.bn_preact = bn_preact
        out.act = act
        out.dropped = dropped

        super(FCLayer, self).__call__()
        return out


class BatchNorm(SameShape):
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

    def __call__(self, x, train_flag=train_flag):
        with tf.name_scope(self.name) as scope:
            if self.called:
                assert rank(x) == self.input_rank
                assert static_size(x, -1) == self.input_shape[-1]
            else:
                self.input_rank = rank(x)
                self.input_shape = x.get_shape()
                self.output_shape = x.get_shape()
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
            super(BatchNorm, self).__call__(normed)
            return normed


class Dropout(SameShape):
    def __init__(self, p_drop, name='Dropout'):
        self.p_drop = p_drop
        super(Dropout, self).__init__(name)

    def __call__(self, x, train_flag=train_flag):

        mask = lambda: tf.to_float(tf.random_uniform(tf.shape(x)) > self.p_drop)
        dropped = tf.cond(train_flag, 
                          lambda: (x * mask()) / (1. - self.p_drop),
                          lambda: x)
        super(Dropout, self).__call__(dropped)
        return dropped


class Stack(Module):
    def __init__(self, layers=None):
        self.layers = []
        if layers:
            [self.add(layer) for layer in layers]
        super(Stack, self).__init__()

    # FIXME: need a better abstraction of passing multiple inputs to this thing
    def __call__(self, x, train_flag=train_flag):
        super(Stack, self).__call__()
        hid = []
        out = x
        for layer in self.layers:
            out = layer(out, train_flag=train_flag)
            hid.append(out)
        out.hid = hid[:-1]
        return out

    def add(self, layer):
        self.layers.append(layer)


class MLP(Stack):
    def __init__(self, sizes, act_fn=tf.nn.relu, bn=False, p_drop=None, readout=True, train_flag=train_flag):
        super(MLP, self).__init__()
        self.n_layer = len(sizes)
        self.n_out = sizes[-1]
        self.output_size = (None, self.n_out)
        self.sizes = sizes

        if bn == False: bn = None
        if p_drop == False: p_drop = None

        self.drop = p_drop is not None
        self.bn = bn is not None
        self.readout = readout

        if type(act_fn) != list:
            act_fn = [act_fn] * self.n_layer
            if self.readout: act_fn[-1] = tf.identity
        self.act_fn = act_fn

        if type(bn) != list:
            bn = [bn] * self.n_layer
            if self.readout: bn[-1] = False
        self.bn = bn

        if type(p_drop) != list:
            p_drop = [p_drop] * self.n_layer
            if self.readout: p_drop[-1] = None
        self.p_drop = p_drop

        for li in range(self.n_layer):
            self.add(FCLayer(self.sizes[li], 
                             self.act_fn[li], 
                             self.bn[li], 
                             self.p_drop[li]))


class lReLU(SameShape):
    def __init__(self, alpha=5.5, name='LReLU'):
        self.alpha = alpha
        super(LReLU, self).__init__()

    def __call__(self, x):
        pos = tf.nn.relu(x)
        neg = alphas * (x - abs(x)) * 0.5
        super(LReLU, self).__call__(x)
        return pos + neg


class pReLU(SameShape):
    def __init__(self, name='LReLU'):
        super(pReLU, self).__init__()

    def __call__(self, x):
        with tf.name_scope(self.name) as scope:
            if self.called:
                # TODO: check shape
                pass
            else:
                self.alpha = tf.Variable(tf.zeros(x.get_shape()[-1], dtype=tf.float32))
            pos = tf.nn.relu(x)
            neg = self.alpha * (x - abs(x)) * 0.5
            super(pReLU, self).__call__(x)
            return pos + neg


if __name__ == '__main__':
    from misc import test_over_mnist
    # TODO: passing a parameterized activation function is clunky
    mlp = MLP([256, 128, 10], act_fn=lambda x: pReLU()(x), bn=True, p_drop=0.2)
    test_over_mnist(mlp, train_flag)

