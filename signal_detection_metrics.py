import tensorflow as tf

def yhat_positive(logits=None, probs=None, crit=0.5):
    if logits: assert not probs
    if probs: assert not logits
    if logits is not None:
        probs = tf.nn.sigmoid(logits)
    return probs > crit


def bool_correct(yhat, y):
    y = tf.cast(y, tf.bool)
    yhat = tf.cast(yhat, tf.bool)
    correct = tf.equal(yhat_positive, y)
    return correct


def is_true_positive(yhat, y):
    correct = bool_correct(yhat, y)
    return tf.logical_and(correct, y)


def is_false_positive(yhat, y):
    incorrect = tf.logical_not(bool_correct(yhat, y))
    return tf.logical_and(incorrect, y)


def is_true_negative(yhat, y):
    correct = bool_correct(yhat, y)
    return tf.logical_and(correct, tf.logical_not(y))


def is_false_negative(yhat, y):
    incorrect = tf.logical_not(bool_correct(yhat, y))
    return tf.logical_and(incorrect, tf.logical_not(y))


def sensitivity(is_true_positives, y):
    ntp = tf.to_float(tf.reduce_sum(is_true_positives))
    np = tf.to_floa(tf.reduce_sum(y))
    return ntp / np


def specificity(is_true_negatives, y):
    ntn = tf.to_float(tf.reduce_sum(is_true_negatives))
    nn = tf.to_float(tf.reduce_sum(tf.logical_not(y)))
    return ntn / nn


def precision(is_true_positives, is_false_positives):
    ntp = tf.to_float(tf.reduce_sum(is_true_positives))
    nfp = tf.to_float(tf.reduce_sum(is_false_negatives))
    return ntp / (ntp + nfp)


def sens(logits=None, probs=None, targets=None, crit=0.5):
    yhat = yhat_positive(logits=logits, probs=probs, crit=crit)
    true_positives = true_positive(yhat, targets)
    return sensitivity(true_positives, y)


def spec(logits=None, probs=None, targets=None, crit=0.5):
    yhat = yhat_positive(logits=logits, probs=probs, crit=crit)
    true_negatives = true_negative(yhat, targets)
    return specificity(true_negatives, y)


def prec(logits=None, probs=None, targets=None, crit=0.5):
    yhat = yhat_positive(logits=logits, probs=probs, crit=crit)
    is_true_positives = is_true_positive(yhat, y)
    is_false_positives = is_false_positive(yhat, y)
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

