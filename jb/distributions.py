import tensorflow as tf

class Distribution(object):
    def __init__(self):
        pass

    def log_prob(self, x):
        pass

    def sample(self, x):
        pass


def log_binomial_coefficient(n, k):
    return tf.lgamma(n+1.) - (tf.lgamma(k+1.) + tf.lgamma(n-k+1.))


class ZeroInflated(Distribution):
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


class NegativeBinomial(Distribution):
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

