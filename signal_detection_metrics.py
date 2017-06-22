import tensorflow as tf

def yhat_positive(logits=None, probs=None, crit=0.5):
    assert not (logits is None and probs is None)
    if logits is not None: assert not probs
    if probs is not None: assert not logits
    if logits is not None:
        probs = tf.nn.sigmoid(logits)
    return probs > crit


def bool_correct(yhat, y):
    y = tf.cast(y, tf.bool)
    yhat = tf.cast(yhat, tf.bool)
    correct = tf.equal(yhat, y)
    return correct


def is_true_positive(yhat, y):
    y = tf.cast(y, tf.bool)
    correct = bool_correct(yhat, y)
    return tf.logical_and(correct, y)


def is_false_positive(yhat, y):
    y = tf.cast(y, tf.bool)
    incorrect = tf.logical_not(bool_correct(yhat, y))
    return tf.logical_and(incorrect, y)


def is_true_negative(yhat, y):
    y = tf.cast(y, tf.bool)
    correct = bool_correct(yhat, y)
    return tf.logical_and(correct, tf.logical_not(y))


def is_false_negative(yhat, y):
    y = tf.cast(y, tf.bool)
    incorrect = tf.logical_not(bool_correct(yhat, y))
    return tf.logical_and(incorrect, tf.logical_not(y))


def sensitivity(is_true_positives, y):
    y = tf.cast(y, tf.bool)
    ntp = tf.reduce_sum(tf.to_float(is_true_positives))
    np = tf.reduce_sum(tf.to_float(y))
    return ntp / np


def specificity(is_true_negatives, y):
    y = tf.cast(y, tf.bool)
    is_true_negatives = tf.to_float(is_true_negatives)
    ntn = tf.reduce_sum(is_true_negatives)
    nn = tf.reduce_sum(tf.to_float(tf.logical_not(y)))
    return ntn / nn


def precision(is_true_positives, is_false_positives):
    is_true_positives = tf.to_float(is_true_positives)
    is_false_positives = tf.to_float(is_false_positives)
    ntp = tf.reduce_sum(is_true_positives)
    nfp = tf.reduce_sum(is_false_negatives)
    return ntp / (ntp + nfp)


def sens(logits=None, probs=None, targets=None, crit=0.5):
    yhat = yhat_positive(logits=logits, probs=probs, crit=crit)
    true_positives = true_positive(yhat, targets)
    return sensitivity(true_positives, y)


def spec(logits=None, probs=None, targets=None, crit=0.5):
    yhat = yhat_positive(logits=logits, probs=probs, crit=crit)
    true_negatives = is_true_negative(yhat, targets)
    return specificity(true_negatives, targets)


def prec(logits=None, probs=None, targets=None, crit=0.5):
    yhat = yhat_positive(logits=logits, probs=probs, crit=crit)
    is_true_positives = is_true_positive(yhat, targets)
    is_false_positives = is_false_positive(yhat, targets)
    return precision(is_true_positives, is_false_positives)


def fpr(logits=None, probs=None, targets=None, crit=0.5):
    return 1 - spec(logits=logits, probs=probs, targets=targets, crit=crit)


def fnr(logits=None, probs=None, targets=None, crit=0.5):
    return 1 - sens(logits=logits, probs=probs, targets=targets, crit=crit)


def fdr(logits=None, probs=None, targets=None, crit=0.5):
    return 1 - prec(logits=logits, probs=probs, targets=targets, crit=crit)


def acc(logits=None, probs=None, targets=None, crit=0.5):
    yhat = yhat_positive(logits=logits, probs=probs, crit=crit)
    correct = bool_correct(yhat, targets)
    return tf.to_float(tf.reduce_sum(correct)) / tf.to_float(tf.shape(targets)[0])

