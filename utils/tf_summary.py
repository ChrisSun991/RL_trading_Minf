import tensorflow as tf


def tf_summary(tag, val):
    """Scalar Value Tensorflow Summary"""
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
