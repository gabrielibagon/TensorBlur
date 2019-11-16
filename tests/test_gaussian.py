import pytest
import numpy as np
import tensorflow as tf

from tensorblur.gaussian import GaussianBlur


def test_create_kernel():
    gauss = GaussianBlur()
    size = 2
    kernel = gauss.create_kernel(size=size, path='../src/tensorblur/coefficients.pkl')
    target = np.array(
        [[[[0.44444448], [0.44444448], [0.44444448]],
          [[0.22222224], [0.22222224], [0.22222224]]],
         [[[0.22222224], [0.22222224], [0.22222224]],
          [[0.11111112], [0.11111112], [0.11111112]]]]
    )
    np.testing.assert_almost_equal(kernel, target)


def test_load_precomputed_coeff():
    path = '../src/tensorblur/coefficients.pkl'
    assert GaussianBlur.load_precomputed_coef(size=-1, path=path) is None

    with pytest.raises(FileNotFoundError):
        GaussianBlur.load_precomputed_coef(size=1, path='')


def test_compute_coeff():
    gauss = GaussianBlur()
    size = 2
    kernel = gauss.compute_coef(size).numpy()
    target = np.array([0.6666667, 0.33333334])
    np.testing.assert_almost_equal(kernel, target)


def test_create_kernel_from_coeff():
    coeff = np.array([0.6666667, 0.33333334])
    kernel = GaussianBlur.create_kernel_from_coeff(coeff).numpy()
    target = np.array(
        [[[[0.44444448], [0.44444448], [0.44444448]],
          [[0.22222224], [0.22222224], [0.22222224]]],
         [[[0.22222224], [0.22222224], [0.22222224]],
          [[0.11111112], [0.11111112], [0.11111112]]]]
    )
    np.testing.assert_almost_equal(kernel, target)


def test_compute_coeffs():
    n = 1
    coef = GaussianBlur.compute_coef(n).numpy()
    assert coef == np.array([1])

    n = 2
    coef = GaussianBlur.compute_coef(n).numpy()
    target = np.array([0.6666667, 0.33333334])
    np.testing.assert_array_almost_equal(coef, target)


