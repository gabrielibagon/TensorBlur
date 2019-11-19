import os
import tensorflow as tf
import numpy as np
import math
import pickle

from tensorblur import utilities


class BlurLayer(tf.keras.layers.Layer):
    def __init__(self, size=None, min_amt=1, max_amt=32):
        super(BlurLayer, self).__init__()
        self.min_amt = min_amt
        self.max_amt = max_amt
        self.kernel_bank = self.precompute_kernels(self.min_amt, self.max_amt)[0]

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the tensor. Format: NHWC

        TODO: hardcoded output shape
        :param input_shape:
        :return:
        """
        return None, 224, 224, 3

    def build(self, input_shape):
        pass

    @staticmethod
    def compute_coeffs(min_amt, max_amt):
        """Compute gaussian coefficients given the size of a kernel"""
        coeffs = {}
        for n in range(min_amt, max_amt+1):
            ceoff = [utilities.binom_coef(n, k) for k in range(n)[::-1]]
            ceoff = tf.divide(ceoff, tf.reduce_sum(ceoff))
            coeffs[n] = ceoff

        return coeffs

    def precompute_kernels(self, min_amt=1, max_amt=32):
        path = 'test_coefficients.pkl'

        if os.path.isfile(path):
            coeffs = pickle.load(open(path, 'rb'))
        else:
            coeffs = self.compute_coeffs(min_amt, max_amt)
        kernels = []
        for size in range(min_amt, max_amt+1):
            kernel = self.create_kernel_from_coeff(coeff=coeffs[size])
            # kernel = utilities.pad_to_width(kernel, width=self.max_amt)
            kernels.append(kernel)
        return kernels

    @staticmethod
    def create_kernel_from_coeff(coeff):
        """
        Generate a gaussian kernel from a list of coefficients
        Kernel must be of shape:
        [filter_height, filter_width, in_channels, channel_multiplier]

        """
        coeff = tf.cast(coeff, tf.float32)
        kernel = tf.einsum('i,j->ij', coeff, coeff)
        kernel = kernel[:, :, tf.newaxis, tf.newaxis]
        kernel = tf.tile(kernel, [1, 1, 3, 1])
        return kernel

    def select_kernels(self, sizes):
        sizes = tf.cast(sizes, tf.int32)
        kernel = tf.gather(self.kernels, sizes)
        kernel = tf.reshape(kernel, shape=(tf.shape(kernel)[0],
                                           tf.shape(kernel)[2], tf.shape(kernel)[3], tf.shape(kernel)[4], tf.shape(kernel)[5]))
        return kernel

    def reshape_kernel(self, kernel):
        batch_size = tf.shape(kernel)[0]
        fh = tf.shape(kernel)[1]
        fw = tf.shape(kernel)[2]
        channels = tf.shape(kernel)[3]
        out_channels = tf.shape(kernel)[4]
        kernel = tf.transpose(kernel, [1, 2, 0, 3, 4])
        kernel = tf.reshape(kernel, [fh, fw, channels * batch_size, out_channels])
        return kernel

    def reshape_input(self, x):
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        channels = tf.shape(x)[3]

        # batch_size, height, width, channels = tf.shape(x)
        inp_r = tf.transpose(x, [1, 2, 0, 3])
        inp_r = tf.reshape(inp_r, [1, height, width, batch_size * channels])
        return inp_r

    def reshape_output(self, out, batch_size, height, width):
        channels = 3
        out = tf.reshape(out, [height, width, batch_size, channels])
        out = tf.transpose(out, [2, 0, 1, 3])
        return out

    def select_random_rotations(self, kernels):
        batch_size = tf.shape(kernels)[0]
        angles = tf.random_uniform(shape=(batch_size,), minval=0, maxval=91, dtype=tf.int32)
        return angles

    @staticmethod
    def format_input(img):
        if len(tf.shape(img)) != 3 and len(tf.shape(img)) != 4:
            num_dims = len(tf.shape(img))
            img_shape = tf.shape(img).numpy()
            raise ValueError('Input image must have shape '
                             '[ batch height width channels ] or [ height width channels ]. '
                             f'Current input has {num_dims} dimensions: {img_shape}')
        elif len(tf.shape(img)) == 3:
            img = tf.expand_dims(img, 0)

        img = tf.cast(img, tf.float32)
        return img

    @staticmethod
    def format_output(img, img_dim):
        img = tf.reshape(img, img_dim)
        return img

    def call(self, inputs, **kwargs):
        if len(self.kernel_bank) > 1:
            kernel = self.kernel_bank
        else:
            kernel = self.kernel_bank[0]

        outputs = tf.nn.depthwise_conv2d(inputs, kernel, strides=[1, 1, 1, 1], padding="SAME")

        return outputs
        # x, amt = inputs[0], inputs[1] #check for inputs etc
        # x = inputs
        # batch_size = tf.shape(x)[0]
        # amt = tf.random.uniform((batch_size, 1), self.min_amt, self.max_amt, tf.int32)
        #
        # batch_size = tf.shape(x)[0]
        # height = tf.shape(x)[1]
        # width = tf.shape(x)[2]
        #
        # kernels = self.select_kernels(amt)
        # angles = self.select_random_rotations(kernels)
        #
        # angles = tf.cast(angles, tf.float32)
        # kernels = kernels[..., 0]
        # kernels, _ = tf.map_fn(lambda x: (tf.contrib.image.rotate(x[0],
        #                                                        tf.multiply(x[1], math.pi / 180),
        #                                                        interpolation='Nearest'), x[1]),
        #                     [kernels, angles],
        #                     dtype=(tf.float64, tf.float32))
        # kernels = kernels[..., None]
        #
        # kernels = self.reshape_kernel(kernels)
        # x = self.reshape_input(x)
        # kernels = tf.cast(kernels, tf.float32)
        # out = tf.nn.depthwise_conv2d(x, filter=kernels, strides=[1, 1, 1, 1], padding='SAME')
        # out = self.reshape_output(out, batch_size, height, width)
        # out = tf.reshape(out, (-1, 224, 224, 3)) # make generic input shape
        # return out

