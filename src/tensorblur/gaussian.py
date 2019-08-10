import os
import pickle
import tensorflow as tf

from tensorblur import utilities
from tensorblur.blur import Blur


class GaussianBlur(Blur):
    """Gaussian Blurring of Images: https://en.wikipedia.org/wiki/Gaussian_blur"""

    def create_kernel(self):
        """Create kernel to apply Gaussian blurring. Use cache if possible"""
        path = 'coefficients.pkl'
        if os.path.isfile(path):
            kernels = self.load_precomputed_kernels()
            if self.size in kernels:
                kernel = kernels[self.size]
            else:
                kernel = self.compute_kernel(self.size)
        else:
            kernel = self.compute_kernel(self.size)
        return kernel

    def load_precomputed_kernels(self, path='coefficients.pkl'):
        """Load kernel from cached coeffiencets on disk"""
        coeffs = pickle.load(open(path, 'rb'))
        kernels = {}
        for idx in range(len(coeffs)):
            kernels[idx+1] = self.create_kernel_from_coeff(coeffs[idx])
        return kernels

    def compute_kernel(self, size: int):
        """Compute a kernel to perform blurring of the specifed size"""
        coeff = self.compute_coefs(size)
        kernel = self.create_kernel_from_coeff(coeff)
        return kernel

    @staticmethod
    def create_kernel_from_coeff(coeff):
        """Generate a gaussian kernel from a list of coefficients"""
        coeff = tf.cast(coeff, tf.float32)
        kernel = tf.einsum('i,j->ij', coeff, coeff)
        kernel = kernel[:, :, tf.newaxis, tf.newaxis]
        kernel = tf.tile(kernel, [1, 1, 3, 1])
        return kernel

    @staticmethod
    def compute_coefs(n):
        """Compute gaussian coefficients given the size of a kernel"""
        coef = [utilities.binom_coef(n, k) for k in range(n)[::-1]]
        coef = tf.divide(coef, tf.reduce_sum(coef))
        return coef
