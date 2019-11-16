import pytest
import numpy as np
import tensorflow as tf

from tensorblur.blur import Blur


def test_create_kernel():
    """Test NotImplementedError in create_kernel"""
    with pytest.raises(NotImplementedError):
        Blur.create_kernel(Blur(), None)


def test_format_input():
    """Test formatting inputs to the appropriate shape"""
    shape = (1,)
    img = np.random.random(shape)
    with pytest.raises(ValueError):
        Blur.format_input(img)

    shape = (1, 1)
    img = np.random.random(shape)
    with pytest.raises(ValueError):
        Blur.format_input(img)

    shape = (1, 1, 1)
    img = np.random.random(shape)
    img = Blur.format_input(img)
    assert len(np.shape(img)) == 4

    shape = (1, 1, 1, 1)
    img = np.random.random(shape)
    img = Blur.format_input(img)
    assert len(np.shape(img)) == 4

    shape = (1, 1, 1, 1)
    img = np.random.random(shape)
    img = tf.cast(img, tf.int32)
    img = Blur.format_input(img)
    assert img.dtype == tf.float32





