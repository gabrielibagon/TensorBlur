import tensorflow as tf


def pad_to_width(kernel, width):
    """
    Zero-pad a kernel to a specific size

    Args:
        kernel:
        width:

    Returns:

    """
    current_width = len(kernel)
    delta = (width - current_width) / 2

    pad0 = tf.cast(delta, tf.int32)
    pad1 = tf.cast(tf.math.ceil(delta), tf.int32)

    paddings = [[pad0, pad1], [pad0, pad1], [0, 0], [0, 0]]

    kernel = tf.pad(kernel, paddings)
    return kernel


def factorial(x):
    """Factorial utility function"""
    return tf.exp(tf.math.lgamma(x+1))


def binom_coef(n, k):
    """
    Compute binomial coefficients of n choose k (i.e. number of ways to choose
    k elements from a set of n elements).

    Args:
        n: size of fixed set
        k: size of subset

    Returns:
        binomial coefficient for n choose k
    """
    n = tf.cast(n, tf.float64)
    k = tf.cast(k, tf.float64)
    num = factorial(n)
    denom = tf.multiply(factorial(k), factorial(n-k))
    return tf.divide(num, denom)