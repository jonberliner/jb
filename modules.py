import tensorflow as tf
from tensorflow.contrib.distributions import Normal, kl
import numpy as np
import pdb

# graph = tf.Graph()
# with graph.as_default():
#     train_flag = tf.placeholder(tf.bool, name='jb_train_flag')
train_flag = tf.placeholder(tf.bool, name='jb_train_flag')

def sess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    return tf.Session(config=config)

def isess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    return tf.InteractiveSession(config=config)


def ph(shape, dtype=tf.float32, name=None):
    return tf.placeholder(dtype=dtype, shape=shape, name=name)


def rank(x):
    return len(x.get_shape())


def static_size(x, d):
    out = x.get_shape()[d].value
    assert out is not None, 'shape of dim d is not static'
    return out


def tf_repeat(tensor, n):
    """behaves like np.repeat, except only runs over the leading dimension n times"""
    out = tf.transpose(tensor)
    rr = rank(tensor)
    tile_size = [n] + [1]*(rr-1)
    out = tf.tile(out, tile_size)

    out = tf.transpose(out)
    reshape_size = tf.concat([[-1], tf.shape(tensor)[1:]], axis=0)
    out = tf.reshape(out, reshape_size)

    # out = tf.reshape(
    #            tf.transpose(tf.tile(tf.transpose(tensor), tf.stack([n, 1]))),
    #            tf.stack([-1, tf.shape(tensor)[1]]))
    return out


class Module(object):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.input_shape = None
        self.output_shape = None
        self.called = False

    def __call__(self, *args, **kwargs):
        with tf.name_scope(self.name) as scope:
            out = self._call(*args, **kwargs)
        self.called = True
        out.call = self
        return out

    def _call(self):
        pass


class SameShape(Module):
    """a module where what comes out is same shape as what went in"""
    def __call__(self, x, **kwargs):
        self.input_rank = rank(x)
        self.input_shape = x.get_shape()
        self.output_shape = x.get_shape()
        return super(SameShape, self).__call__(x, **kwargs)


class Linear(Module):
    def __init__(self, n_out, W_init='msra', name='Linear'):
        super(Linear, self).__init__(name)

        self.n_out = n_out
        self.n_in = None
        self.output_shape = (None, n_out)
        self.W_init = W_init

    def _call(self, x):
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

        out = tf.matmul(x, self.W) + self.b
        return out


def _assign_moving_average(orig_val, new_val, decay, name='_assign_moving_average'):
    with tf.name_scope(name):
        td = decay * (new_val - orig_val)
        return tf.assign_add(orig_val, td)


class BatchNorm(SameShape):
    def __init__(self, loc_trainable=True, scale_trainable=True, decay=0.5, eps=1e-3, add_noise=False, name='BatchNorm'):
        """
            loc_trainable: bool, learnable means
            scale_trainable: bool, learnable vars
            decay: scalar in (0., 1.), how fast moving average updates (bigger is faster)
            eps: scalar in (0., inf), stability term for batch var
            name:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        super(BatchNorm, self).__init__(name)

        self.loc_trainable = loc_trainable
        self.scale_trainable = scale_trainable
        self.decay = decay
        self.eps = eps
        self.add_noise = add_noise

    def _call(self, x, train_flag=train_flag):
        if self.called:
            assert rank(x) == self.input_rank
            assert static_size(x, -1) == self.input_shape[-1]
        else:
            self.pool_axes = np.arange(self.input_rank-1).tolist()
            self.n_out = static_size(x, -1)
            self.moving_mean = tf.Variable(tf.constant(0.0, shape=[self.n_out]),
                                        name='moving_mean', trainable=False)
            self.moving_var = tf.Variable(tf.constant(1.0, shape=[self.n_out]),
                                        name='moving_var', trainable=False)

            self.loc = tf.Variable(tf.constant(0.0, shape=[self.n_out]),
                                        name='beta', trainable=self.loc_trainable)
            self.scale = tf.nn.softplus(tf.Variable(tf.constant(1.0, shape=[self.n_out]),
                                            name='gamma', trainable=self.scale_trainable))


        batch_mean, batch_var = tf.nn.moments(x, self.pool_axes, name='moments')
        def training():
            update_mm = _assign_moving_average(self.moving_mean, batch_mean, self.decay)
            update_mv = _assign_moving_average(self.moving_var, batch_var, self.decay)
            with tf.control_dependencies([update_mm, update_mv]):
                normed = tf.nn.batch_normalization(x, batch_mean, batch_var, self.loc, self.scale, self.eps)
                if self.add_noise:
                    normed += Normal(self.loc, self.scale).sample(tf.shape(normed)[0])
                return normed

        def testing():
            return tf.nn.batch_normalization(x, self.moving_mean, self.moving_var, self.loc, self.scale, self.eps)

        normed = tf.cond(train_flag, training, testing)

        normed.batch_mean = batch_mean
        normed.batch_var = batch_var
        return normed


class Dropout(SameShape):
    def __init__(self, p_drop, name='Dropout'):
        super(Dropout, self).__init__(name)

        self.p_drop = p_drop

    def _call(self, x, train_flag=train_flag):
        mask = lambda: tf.to_float(tf.random_uniform(tf.shape(x)) > self.p_drop)
        dropped = tf.cond(train_flag,
                          lambda: (x * mask()) / (1. - self.p_drop),
                          lambda: x)
        return dropped


class FCLayer(Module):
    # TODO: decide if we want to break into containers, stacks, etc
    def __init__(self, n_out, act_fn=tf.nn.relu, bn=False, p_drop=False, bn_noise=False, name='FCLayer'):
        super(FCLayer, self).__init__(name)

        with tf.name_scope(self.name) as scope:
            self.n_out = n_out
            self.bn = bn
            self.bn_noise = bn_noise
            self.p_drop = p_drop
            self.drop = p_drop is not None

            self.linear = Linear(n_out)
            self.batch_norm = BatchNorm(add_noise=self.bn_noise) if self.bn else None
            self.act_fn = act_fn
            self.dropout = Dropout(p_drop) if self.drop else None

    def _call(self, x, train_flag=train_flag):
        if self.called:
            assert rank(x) == self.input_rank
            assert static_size(x, 1) == self.n_in
        else:
            self.input_rank = rank(x)
            self.input_shape = x.get_shape()
            self.n_in = static_size(x, 1)

        preact = self.linear(x)
        bn_preact = self.batch_norm(preact, train_flag=train_flag) if self.bn else None
        act = self.act_fn(bn_preact) if self.bn else self.act_fn(preact)
        dropped = self.dropout(act, train_flag=train_flag) if self.drop else None

        out = dropped if self.drop else act
        out.preact = preact
        out.bn_preact = bn_preact
        out.act = act
        out.dropped = dropped

        return out


class Conv2D(Module):
    # TODO: decide if we want to break into containers, stacks, etc
    def __init__(self, n_out, hw, stride=None, padding='SAME', W_init='mrsa', name='Conv2D'):
        super(Conv2D, self).__init__(name)
        self.n_out = n_out
        self.n_in = None
        self.output_shape = (None, None, None, n_out)
        self.W_init = W_init
        self.padding = padding
        if type(hw) != list: hw = [hw]*2
        assert len(hw) == 2
        self.hw = hw
        self.kh =  hw[0]
        self.kw =  hw[1]
        if type(stride) != list: stride = [stride]*2
        stride = stride or [1, 1]
        assert len(stride) == 2
        self.stride = stride
        self.padding = padding

    def _call(self, x):
        if self.called:
            assert rank(x) == self.input_rank
            assert static_size(x, 1) == self.n_in
        else:
            self.input_shape = x.get_shape()
            self.n_in = static_size(x, 1)
            if self.W_init == 'msra':
                W0 = tf.random_normal([self.kh, self.kw, self.n_in, self.n_out]) *\
                                     (tf.sqrt(2. / (self.n_in*self.kh*self.kw)))
            else:
                raise ValueError('W_init must be in ["msra"]')
            self.W = tf.Variable(W0, name='W')
            self.b = tf.Variable(tf.zeros([self.n_out]), name='b')

        return tf.nn.conv2d(x, self.W, self.strides, self.padding) + self.b


class Conv2DLayer(Module):
    # TODO: decide if we want to break into containers, stacks, etc
    def __init__(self, n_out, kh, kw, stride, padding='SAME', W_init='mrsa', act_fn=tf.nn.relu, bn=False, p_drop=False, name='Conv2DLayer'):
        super(Conv2DLayer, self).__init__(name)

        with tf.name_scope(self.name) as scope:
            self.n_out = n_out

            self.bn = bn
            self.p_drop = p_drop
            self.drop = p_drop is not None

            self.conv2d = Conv2D(n_out, kh, kw, stride, padding, W_init)
            self.batch_norm = BatchNorm() if self.bn else None
            self.act_fn = act_fn
            self.dropout = Dropout(p_drop) if self.drop else None

    def _call(self, x, train_flag=train_flag):
        if self.called:
            assert rank(x) == self.input_rank
            assert static_size(x, 1) == self.n_in
        else:
            self.input_rank = rank(x)
            self.input_shape = x.get_shape()
            self.n_in = static_size(x, 1)

        preact = self.conv2d(x)
        bn_preact = self.batch_norm(preact, train_flag=train_flag) if self.bn else None
        act = self.act_fn(bn_preact) if self.bn else self.act_fn(preact)
        dropped = self.dropout(act, train_flag=train_flag) if self.drop else None

        out = dropped if self.drop else act
        out.preact = preact
        out.bn_preact = bn_preact
        out.act = act
        out.dropped = dropped

        return out


class Stack(Module):
    def __init__(self, layers=None, name='Stack'):
        super(Stack, self).__init__(name)

        self.layers = []
        if layers:
            [self.add(layer) for layer in layers]

    # FIXME: need a better abstraction of passing multiple inputs to this thing
    def _call(self, x, **kwargs):
        hid = []
        out = x
        for layer in self.layers:
            out = layer(out, **kwargs)
            hid.append(out)
        out.hid = hid[:-1]
        return out

    def add(self, layer):
        self.layers.append(layer)


class MLP(Stack):
    def __init__(self, sizes, act_fn=tf.nn.relu, bn=False, bn_noise=False, p_drop=None, readout=True, name='MLP'):
        super(MLP, self).__init__(name=name)

        self.n_layer = len(sizes)
        self.n_out = sizes[-1]
        self.output_shape = (None, self.n_out)
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

        if type(bn_noise) != list:
            bn_noise = [bn_noise] * self.n_layer
            if self.readout: bn_noise[-1] = False
        self.bn_noise = bn_noise

        if type(p_drop) != list:
            p_drop = [p_drop] * self.n_layer
            if self.readout: p_drop[-1] = None
        self.p_drop = p_drop

        with tf.name_scope(self.name) as scope:
            for li in range(self.n_layer):
                self.add(FCLayer(self.sizes[li],
                                self.act_fn[li],
                                self.bn[li],
                                self.p_drop[li],
                                self.bn_noise[li],
                                name='FCLayer_%d' % (li)))


class Cartesian(Module):
    def __init__(self, reverse_order=False, name='Cartesian'):
        super(Cartesian, self).__init__(name)
        self.reverse_order = reverse_order

    def _call(self, s0=None, s1=None):
        assert (s0 is not None) or (s1 is not None), 's0 and s1 cannot both be None'
        if s1 is None and s0 is None:
            return None
        elif s1 is None:
            return s0
        elif s0 is None:
            return s1
        else:
            # TODO: should be able to work with any rank
            assert rank(s0) <= 2, 'currently, both sets must be rank <= 2'
            assert rank(s1) <= 2, 'currently, both sets must be rank <= 2'
            ns0 = tf.shape(s0)[0]
            ns1 = tf.shape(s1)[0]
            assert rank(s0) == rank(s1), 's0 and s1 must be same rank'
            sets_rank = rank(s0)

            if sets_rank == 0:
                return tf.expand_dims(tf.stack([s0, s1]), 1)

            if sets_rank == 1:
                s0 = tf.expand_dims(s0, 1)
                s1 = tf.expand_dims(s1, 1)
                sets_rank += 1

            if self.reverse_order:
                s0_tile_size = [ns1] + [1]*(sets_rank-1)
                rs0 = tf.tile(s0, s0_tile_size)
                rs1 = tf_repeat(s1, ns0)
            else:
                rs0 = tf_repeat(s0, ns1)
                s1_tile_size = [ns0] + [1]*(sets_rank-1)
                rs1 = tf.tile(s1, s1_tile_size)

            out = tf.concat([rs0, rs1], 1)
            ds0 = static_size(s0, 1)
            ds1 = static_size(s1,1)
            out.set_shape([None] + [ds0+ds1])
            return out

# def tf_repeat(tensor, n):
#     """behaves like np.repeat, except only runs over the leading dimension n times"""
#     out = tf.transpose(tensor)
#     rr = rank(tensor)
#     tile_size = [n] + [1]*(rr-1)
#     out = tf.tile(out, tile_size)

#     out = tf.transpose(out)
#     reshape_size = tf.concat([[-1], tf.shape(tensor)[1:]], axis=0)
#     out = tf.reshape(out, reshape_size)

#     # out = tf.reshape(
#     #            tf.transpose(tf.tile(tf.transpose(tensor), tf.stack([n, 1]))),
#     #            tf.stack([-1, tf.shape(tensor)[1]]))
#     return out


class lReLU(SameShape):
    def __init__(self, alpha=5.5, name='lReLU'):
        super(lReLU, self).__init__(name)

        self.alpha = alpha

    def _call(self, x):
        pos = tf.nn.relu(x)
        neg = self.alpha * (x - abs(x)) * 0.5
        return pos + neg


class pReLU(SameShape):
    def __init__(self, name='pReLU'):
        super(pReLU, self).__init__(name)

    def _call(self, x):
        if self.called:
            # TODO: check shape
            pass
        else:
            self.alpha = tf.Variable(tf.zeros(x.get_shape()[-1], dtype=tf.float32))
        pos = tf.nn.relu(x)
        neg = self.alpha * (x - abs(x)) * 0.5
        return pos + neg


if __name__ == '__main__':
    from misc import test_over_mnist
    # TODO: passing a parameterized activation function is clunky
    mlp = MLP([256, 128, 10], act_fn=lambda x: pReLU()(x), bn=True, p_drop=0.2)
    test_over_mnist(mlp, train_flag)

