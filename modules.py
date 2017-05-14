import tensorflow as tf
from tensorflow.contrib.distributions import Normal, kl
import numpy as np
import pdb

# from https://stats.stackexchange.com/questions/228704/how-does-one-initialize-neural-networks-as-suggested-by-saxe-et-al-using-orthogo
def orthogonal_initializer(scale = 1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    print('Warning -- You have opted to use the orthogonal_initializer function')
    def _initializer(shape, dtype=tf.float32):
      flat_shape = (shape[0], np.prod(shape[1:]))
      a = np.random.normal(0.0, 1.0, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape) #this needs to be corrected to float32
      print('you have initialized one orthogonal matrix.')
      return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer


def batch_norm(n_out, input_rank, phase_train, beta_trainable=True, gamma_trainable=True, scope='bn'):
    """
        n_out:       integer, depth of input maps
        input_rank:  rank of input x
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                    name='beta', trainable=beta_trainable)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                    name='gamma', trainable=gamma_trainable)
        pool_axes = np.arange(input_rank-1).tolist()
    def call(x):
        """
            Batch normalization on convolutional maps.
            Args:
                x:           Tensor, 4D BHWD input maps
        """
        with tf.variable_scope(scope):
            batch_mean, batch_var = tf.nn.moments(x, pool_axes, name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

            setattr(normed, 'batch_mean', batch_mean)
            setattr(normed, 'batch_var', batch_var)
            setattr(normed, 'mean', mean)
            setattr(normed, 'var', var)
            setattr(normed, 'call', call)
        return normed

    setattr(call, 'beta', beta)
    setattr(call, 'gamma', gamma)
    return call


def linear(n_in, n_out):
    W = tf.Variable(tf.random_normal([n_in, n_out]) * np.sqrt(2./n_in))
    b = tf.Variable(tf.zeros([n_out]))

    def call(x):
        out = tf.matmul(x, W) + b
        setattr(out, 'call', call)
        return out

    setattr(call, 'W', W)
    setattr(call, 'b', b)
    return call


def MixtureOfGaussians(k):
    mu = tf.Variable(tf.random_normal([k]))
    log_var = tf.Variable(tf.random_normal([k]))
    w = tf.exp(tf.Variable(tf.random_normal([k])))

    components = Normal(mu, tf.sqrt(tf.exp(log_var)))

    def pdf(x):
        assert len(x.get_shape()) == 1, 'x should be vector of mu_hat'
        x = tf.tile(tf.expand_dims(x, 1), [1, k])
        unweighted_pdfs = components.pdf(x)
        weighted_pdfs = (unweighted_pdfs * w) / tf.reduce_sum(w)
        out = tf.reduce_sum(weighted_pdfs, [1])
        return out

    def log_pdf(x):
        return tf.log(pdf(x))

    return call


def lrelu(_x, alphas=5.5, name='lrelu'):
    with tf.name_scope(name) as scope:
        alphas = tf.Variable(alphas, dtype=tf.float32, trainable=False, name='alpha')
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5
        return pos + neg


def prelu(_x):
  alphas = tf.Variable(tf.zeros(_x.get_shape()[-1], dtype=tf.float32))
  # alphas = tf.get_variable('alpha', _x.get_shape()[-1],
  #                      initializer=tf.constant_initializer(0.0),
  #                      dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5
  return pos + neg


def mlp(sizes, act_fn=tf.nn.relu, bn=None):
    linears = [linear(sizes[li], sizes[li+1]) for li in range(len(sizes)-1)]

    def call(x):
        activations = []
        bns = [None] * len(sizes)
        out = x
        activations.append(out)
        for li, layer in enumerate(linears[:-1]):
            out = layer(out)
            activations.append(out)
            if bn is not None:
                out = batch_norm(out.get_shape()[1], 2, bn)(out)
                bns[li] = out
            out = act_fn(out)
        out = linears[-1](out)
        activations.append(out)

        setattr(out, 'activations', activations)
        setattr(out, 'bns', bns)
        setattr(out, 'call', call)
        return out

    setattr(call, 'linears', linears)
    setattr(call, 'act_fn', act_fn)
    return call


def sample_poisson(rate):
    L = tf.exp(-tf.constant(rate))
    k = tf.constant(0.)
    p = tf.constant(1.)
    cond = lambda k, p: tf.greater(p, L)
    def inc(k, p):
        k = tf.add(k, 1.)
        p = tf.multiply(p, tf.random_uniform(()))
        return k, p
    kk, pp = tf.while_loop(cond, inc, [k, p])
    return kk - 1.


def static_size(x, d):
    out = x.get_shape()[d].value
    assert out is not None, 'shape of dim d is not static'
    return out


def tf_repeat(tensor, n):
    """behaves like np.repeat, except only runs over the leading dimension n times"""
    out = tf.reshape(
               tf.transpose(tf.tile(tf.transpose(tensor), tf.stack([n, 1]))),
               tf.stack([-1, tf.shape(tensor)[1]]))
    return out


def cartesian(reverse_order=False):
    def call(s0=None, s1=None):
        assert (s0 is not None) or (s1 is not None), 's0 and s1 cannot both be None'
        if s1 is None and s0 is None:
            return None
        elif s1 is None:
            return s0
        elif s0 is None:
            return s1
        else:
            with tf.name_scope('cartesian') as scope:
                ns0 = tf.shape(s0)[0]
                ns1 = tf.shape(s1)[0]
                if reverse_order:
                    rs0 = tf.tile(s0, tf.stack([ns1,1]))
                    rs1 = tf_repeat(s1, ns0)
                else:
                    rs0 = tf_repeat(s0, ns1)
                    rs1 = tf.tile(s1, tf.stack([ns0,1]))

                out = tf.concat([rs0, rs1], 1)
                ds0 = static_size(s0,1)
                ds1 = static_size(s1,1)
                out.set_shape((None, ds0+ds1))
            return out
    return call


def log_binomial_coefficient(n, k):
    return tf.lgamma(n+1.) - (tf.lgamma(k+1.) + tf.lgamma(n-k+1.))


class ZeroInflated(object):
    def __init__(self, distribution, phi):
        self.distribution = distribution
        self.phi = phi
        self.mean = lambda: (1.-self.phi) * self.distribution.mean()
        def mode():
            dmode = self.distribution.mode()
            pmf_dmode = tf.exp(self.distribution.log_pmf(dmode)) * (1.-self.phi)
            return tf.cast(self.phi < pmf_dmode, tf.float32) * pmf_dmode
        self.mode = mode

    def log_pmf(self, k, STABILITY=1e-6):
        d_log_pmf = self.distribution.log_pmf(k)
        is_zero = tf.cast(tf.equal(k, 0.), tf.float32)
        out = tf.log(((is_zero * self.phi) + tf.exp(d_log_pmf) * (1.-self.phi))+STABILITY)
        return out

    log_prob = log_pmf


class NegativeBinomial(object):
    def __init__(self, r, p):
        self.r = r
        self.p = p
        self.shape = tf.shape(r)
        self.mean = lambda: (p*r) / (1.-p)
        self.mode = lambda: tf.floor((p*(r-1.)) / (1.-p)) * tf.cast(r > 0., tf.float32)
        self.variance = lambda: (p*r) / tf.square(1.-p)

    def log_pmf(self, k):
        # FIXME: atm k must be same size as r and p
        out = log_binomial_coefficient(k + self.r - 1., k) +\
               self.r * tf.log(1.-self.p) +\
               k * tf.log(self.p)
        return out

    log_prob = log_pmf



def hard_sigmoid(x, STABILITY=1e-6):
    return tf.clip_by_value(x*0.2 + 0.5, 0.+STABILITY, 1.-STABILITY)


def vae(dz, encoder, decoder):
    def call(x):
        # x_em = linear(x)
        # z_em = encoder(x_em)
        qz_em = encoder(x)
        qz_mlv = linear(static_size(qz_em, 1), dz*2)(qz_em)
        qz_mu, qz_lv = tf.unstack(tf.reshape(qz_mlv, [-1, dz, 2]), axis=2)
        qz_var = tf.nn.softplus(qz_lv)

        z = Normal(qz_mu, qz_lv).sample()

        px = decoder(z)
        px.z = z
        px.qz_mu = qz_mu
        px.qz_var = qz_var

        px.kl_z_qp = kl(Normal(qz_mu, qz_var), Normal(0., 1.))

        px.call = call
        return px
    call.encoder = encoder
    call.decoder = decoder
    return call


def dropout(p_keep, train_flag):
    def call(x):
        out = tf.cond(train_flag,
                      lambda: (x * tf.cast(tf.random_uniform(tf.shape(x)) < p_keep, tf.float32)) / p_keep,
                      lambda: x)
        out.call = call
        return out
    return call


def fc_layer(n_in, n_out, act_fn=tf.identity, bn=False, p_keep=None, train_flag=None):
    if bn or p_keep: assert train_flag is not None

    lin = linear(n_in, n_out)
    bn = batch_norm(n_out, 2, train_flag) if bn else tf.identity
    drop = dropout(p_keep, train_flag) if p_keep else tf.identity

    def call(x):
        out = pre_act = lin(x)
        out = bn_pre_act = bn(out)
        out = act = act_fn(out)
        out = _drop = drop(out)

        out.pre_act = pre_act
        out.bn_pre_act = bn_pre_act
        out.act = act
        out.drop = _drop
        out.call = call
        return out

    call.lin = lin
    call.bn = bn
    call.act_fn = act_fn
    call.drop = drop
    return call


