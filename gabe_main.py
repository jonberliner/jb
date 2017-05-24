import jb.modules as jbm
import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import numpy as np
import jb.util as jbu
import pdb


def conv2d(x, ch_out, khw, stride, padding):
    stride = [1, stride, stride, 1]
    kh, kw = khw
    n_in = jbm.static_size(x, 3)
    W0 = tf.random_normal([kh, kw, ch_in, ch_out]) *\
                         (tf.sqrt(2. / (ch_in*kh*kw)))

    W = tf.Variable(W0, name='W')
    b = tf.Variable(tf.zeros([ch_out]), name='b')

    out = tf.nn.conv2d(x, W, stride, padding) + b
    out.W = W
    out.b = b

    return out


def layer(out0, ch_out, padding='SAME'):
    out = conv2d(out0, ch_out, khw=[3, 3], stride=1, padding=padding)
    out = jbm.BatchNorm()(out)
    out = tf.nn.relu(out)
    out = conv2d(out0, ch_out, khw=[3, 3], stride=1, padding=padding)
    return out


# DOWN-SAMPLING DECISION: we're doing strided conv for downsampling.  Can also try max pooling.
def stride(out0, ch, padding='SAME'):
    return conv2d(out0, ch, khw=[3,3], stride=2, padding=padding)

POOL = stride

def stride_encoder(x, train_flag=jbm.train_flag):
    #FIXME:
    gabe_fixed = False  # NOTE: Gabe - get rid of this when you change to fit your data sizes.  xojb
    assert gabe_fixed, 'change this to fit your data!'
    out = tf.reshape(x, [-1, 28, 28, 1])
    out = layer(out, 8)
    out = POOL(out, 8)  # 14 x 14
    out = layer(out, 16)
    out = POOL(out, 16)  # 7 x 7
    out = layer(out, 32)
    out = POOL(out, 32)  # 3 x 3
    out = layer(out, 64, padding='VALID')
    # READOUT DECISION: we're doing spatial pooling.  Can also try flattening, end2end conv, etc.
    out = tf.reduce_sum(out, [1,2])
    out = jbm.Linear(10)(out)
    # out = tf.squeeze(out, axis=[1,2])
    return out


regular_mlp = jbm.MLP([256, 256, 10], bn=True, p_drop=0.2)
if __name__ == '__main__':
    MODEL = stride_encoder
    # from jb.misc import test_over_mnist, test_over_data
    gabe_fixed = False  # NOTE: Gabe - get rid of this when you change dat to your dataset.  xojb
    assert gabe_fixed, 'change this to fit your data!'
    from tensorflow.examples.tutorials.mnist import input_data
    dat = input_data.read_data_sets("MNIST_data/", one_hot=True)
    test_over_data(dat, model=MODEL, train_flag_ph=jbm.train_flag, BS=64, LR=1e-2)

