import os
import pickle
import tensorflow as tf

from tensorblur import utilities
from tensorblur.blur import Blur


class GaussianBlur(Blur):
    """
    Gaussian Blurring of Images: https://en.wikipedia.org/wiki/Gaussian_blur

    See base class `Blur` in `blur.py`
    """

    def create_kernel(self, size=1, path='coefficients.pkl'):
        """
        Create gaussian kernel.

        Args:
            size: pixel height and width of a square kernel

        Returns:
            kernel of shape [filter_height, filter_width, in_channels, channel_multiplier]
        """
        coeff = None

        # Load cached kernel if possible
        if os.path.isfile(path):
            coeff = self.load_precomputed_coeff(size=size, path=path)

        # If cache does not exist (or if empty file), compute coefficients
        if coeff is None:
            coeff = self.compute_coeff(size)

        # format coefficients to square kernel
        kernel = self.create_kernel_from_coeff(coeff)

        return kernel

    @staticmethod
    def load_precomputed_coeff(size=1, path='coefficients.pkl'):
        """
        Load kernel from cached coeffients on disk

        Args:
            size: kernel size
            path: path to coefficient cache

        Returns:
            coefficients of a particular kernel size
        """
        coeffs = pickle.load(open(path, 'rb'))

        if size in coeffs:
            return coeffs[size]
        else:
            return None

    @staticmethod
    def create_kernel_from_coeff(coeff):
        """
        Generate a gaussian kernel from a list of coefficients
        Args:
            coeff: list of coefficients of a particular kernel size

        Returns:
            kernel of shape [filter_height, filter_width, in_channels, channel_multiplier]
        """
        coeff = tf.cast(coeff, tf.float32)
        kernel = tf.einsum('i,j->ij', coeff, coeff)
        kernel = kernel[:, :, tf.newaxis, tf.newaxis]
        kernel = tf.tile(kernel, [1, 1, 3, 1])
        return kernel

    @staticmethod
    def compute_coeff(size):
        """
        Compute Gaussian coefficients given the size of a kernel. Uses binomial approach.
        
        Args:
            size: kernel size

        Returns:
            list of coefficients
        """
        coef = [utilities.binom_coef(size, k) for k in range(size)[::-1]]
        coef = tf.divide(coef, tf.reduce_sum(coef))
        return coef
