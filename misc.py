from __future__ import print_function
import tensorflow as tf
import numpy as np
import jb.util as jbu
import jb.modules as jbm


def test_over_mnist(model, train_flag_ph=None, linear_read_in=False, linear_read_out=False):
    # TODO: make read in/out magic available
    assert linear_read_in == False
    assert linear_read_out == False

    from tensorflow.examples.tutorials.mnist import input_data
    seed = None
    rng = np.random.RandomState(seed)
    dat = input_data.read_data_sets("MNIST_data/", one_hot=True)
    DX = dat.train.images.shape[1]
    DY = dat.train.labels.shape[1]

    if train_flag_ph is None:
        train_flag_ph = tf.placeholder(tf.bool)
    x_ph = tf.placeholder(tf.float32, (None, DX))
    y_ph = tf.placeholder(tf.float32, (None, DY))

    # FIXME: no good train flag solution yet
    yhat_logit = model(x_ph, train_flag_ph)
    yhat = tf.nn.softmax(yhat_logit)

    losses = tf.nn.softmax_cross_entropy_with_logits(logits=yhat_logit, labels=y_ph)
    loss = tf.reduce_sum(losses)

    trainer = tf.train.AdamOptimizer(1e-3).minimize(loss)

    # start training
    BS = 64
    batcher = jbu.Batcher(dat.train.num_examples, BS)

    N_STEP = int(1e3)
    TEST_EVERY = int(1e2)

    def prep_fd(d0, i_batch, training):
        if i_batch is not None:
            x0 = d0.images[i_batch]
            y0 = d0.labels[i_batch]
        else:
            x0 = d0.images
            y0 = d0.labels

        out = {x_ph: x0, y_ph: y0, train_flag_ph: training}
        return out


    def train(sess, d0, batcher):
        i_batch = batcher()
        fd = prep_fd(d0, i_batch, True)
        _, loss0 = sess.run([trainer, loss], feed_dict=fd)


    def test(d0, name, i_step):
        n0 = d0.images.shape[0]
        i_start = 0
        l0 = 0.
        p_correct = 0.
        yh = []
        while i_start < n0:
            i_end = min(i_start + BS, n0)
            fd = prep_fd(d0, np.arange(i_start, i_end), False)
            l0 += sess.run(loss, feed_dict=fd)
            yh0 = sess.run(yhat, feed_dict=fd)
            correct = np.equal(np.argmax(yh0, 1), np.argmax(fd[y_ph], 1))
            p_correct += correct.sum()
            yh.append(yh0)
            i_start = i_end
        y0 = d0.labels

        p_correct /= n0
        print('%s acc step %d: %03f' % (name, i_step, p_correct))

        l0 /= n0
        print('%s loss step %d: %03f' % (name, i_step, l0))


    with jbm.sess() as sess:
        tf.global_variables_initializer().run()
        for i_step in range(N_STEP):
            if i_step % TEST_EVERY == 0:
                test(dat.train, 'train', i_step)
                test(dat.test, 'test', i_step)
            train(sess, dat.train, batcher)


if __name__ == '__main__':
    def test_model(x, train_flag_ph):
        DY = 10
        model = jbm.MLP([256, 256, DY], bn=True, p_drop=0.2)
        return yhat_logit

    test_over_mnist(test_model)
