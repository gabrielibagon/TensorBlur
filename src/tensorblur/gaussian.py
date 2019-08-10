import os
import pickle
import tensorflow as tf

from tensorblur import utilities
from tensorblur.blur import Blur


class GaussianBlur(Blur):
    """Gaussian Blurring of Images: https://en.wikipedia.org/wiki/Gaussian_blur"""

    def create_kernel(self, size=1, path='coefficients.pkl'):
        """Create kernel to apply Gaussian blurring. Use cache if possible"""

        coeff = None

        if os.path.isfile(path):
            coeff = self.load_precomputed_coeff(size=size, path=path)

        if coeff is None:
            coeff = self.compute_coeff(size)

        kernel = self.create_kernel_from_coeff(coeff)
        return kernel

    @staticmethod
    def load_precomputed_coeff(size=1, path='coefficients.pkl'):
        """Load kernel from cached coeffiencets on disk"""
        coeffs = pickle.load(open(path, 'rb'))

        if size in coeffs:
            return coeffs[size]
        else:
            return None

    @staticmethod
    def create_kernel_from_coeff(coeff):
        """Generate a gaussian kernel from a list of coefficients"""
        coeff = tf.cast(coeff, tf.float32)
        kernel = tf.einsum('i,j->ij', coeff, coeff)
        kernel = kernel[:, :, tf.newaxis, tf.newaxis]
        kernel = tf.tile(kernel, [1, 1, 3, 1])
        return kernel

    @staticmethod
    def compute_coeff(n):
        """Compute gaussian coefficients given the size of a kernel"""
        coef = [utilities.binom_coef(n, k) for k in range(n)[::-1]]
        coef = tf.divide(coef, tf.reduce_sum(coef))
        return coef
