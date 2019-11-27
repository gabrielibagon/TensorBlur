import math
import numpy as np
import tensorflow as tf
from scipy.ndimage.interpolation import rotate

from tensorblur import utilities


class RandomBlur(GaussianBlur):
    def __init__(self, path=''):
        path = '/home/gabe/TensorBlur/src/tensorblur/test_coefficients.pkl'
        self.coefs = self.load_precomputed_kernel(path)

    def select_random_kernels(self, batch_size):
        idx = np.random.randint(1, len(self.coefs), size=batch_size)
        kernel = tf.stack([self.coefs[i] for i in idx])
        return kernel, idx

    def reshape_kernel(self, kernel):
        batch_size, fh, fw, channels, out_channels = tf.shape(kernel)
        kernel = tf.transpose(kernel, [1, 2, 0, 3, 4])
        kernel = tf.reshape(kernel, [fh, fw, channels * batch_size, out_channels])
        return kernel

    def reshape_input(self, x):
        batch_size, height, width, channels = tf.shape(x)
        inp_r = tf.transpose(x, [1, 2, 0, 3])
        inp_r = tf.reshape(inp_r, [1, height, width, batch_size * channels])
        return inp_r

    def reshape_output(self, out, batch_size, height, width):
        channels = 3
        out = tf.reshape(out, [height, width, batch_size, channels])
        out = tf.transpose(out, [2, 0, 1, 3])
        return out

    def __call__(self, x):
        batch_size, height, width, channels = tf.shape(x)
        kernel, amounts = self.select_random_kernels(batch_size)
        kernel = self.reshape_kernel(kernel)
        x = self.reshape_input(x)
        out = tf.nn.depthwise_conv2d(x, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
        out = self.reshape_output(out, batch_size, height, width)

        return out, amounts


class AverageBlur:
    def __init__(self, size=3):
        self.size = size
        self.kernel = self.create_kernel(size)

    @staticmethod
    def create_kernel(size):
        return tf.ones((size, size, 3, 1))/(size**2)

    def __call__(self, img, amount=1):
        img = tf.cast(img, tf.float32)
        img = tf.nn.depthwise_conv2d(img, self.kernel, strides=[1, 1, 1, 1], padding="SAME")
        return img


class BlurLayer(tf.keras.layers.Layer):
    def __init__(self, min_amt=1, max_amt=32):
        super(BlurLayer, self).__init__()
        self.num_channels = 3
        self.min_amt = min_amt
        self.max_amt = max_amt
        self.kernels = self.precompute_kernels(self.min_amt, self.max_amt)

    def compute_output_shape(self, input_shape):
        return None, 224, 224, 3


    def build(self, input_shape):
        pass

    def precompute_kernels(self, min_amt=1, max_amt=32):
        kernels = []
        for size in range(min_amt, max_amt+1):
            kernel = self.create_kernel(size=size)
            kernel = utilities.pad_to_width(kernel, width=self.max_amt)
            kernels.append(kernel)
        return kernels

    def create_kernel(self, size):
        kernel = np.zeros((size, size, self.num_channels, 1))
        kernel[:, np.shape(kernel)[1] // 2] = 1.
        kernel /= (np.sum(kernel)/self.num_channels)
        return kernel

    def select_kernels(self, amt):
        amt = tf.cast(amt, tf.int32)
        kernel = tf.gather(self.kernels, amt)
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

    def call(self, inputs, **kwargs):
        # x, amt = inputs[0], inputs[1] #check for inputs etc
        x = inputs
        batch_size = tf.shape(x)[0]
        amt = tf.random.uniform((batch_size, 1), self.min_amt, self.max_amt, tf.int32)

        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]

        kernels = self.select_kernels(amt)
        angles = self.select_random_rotations(kernels)

        angles = tf.cast(angles, tf.float32)
        kernels = kernels[..., 0]
        kernels, _ = tf.map_fn(lambda x: (tf.contrib.image.rotate(x[0],
                                                               tf.multiply(x[1], math.pi / 180),
                                                               interpolation='Nearest'), x[1]),
                            [kernels, angles],
                            dtype=(tf.float64, tf.float32))
        kernels = kernels[..., None]

        kernels = self.reshape_kernel(kernels)
        x = self.reshape_input(x)
        kernels = tf.cast(kernels, tf.float32)
        out = tf.nn.depthwise_conv2d(x, filter=kernels, strides=[1, 1, 1, 1], padding='SAME')
        out = self.reshape_output(out, batch_size, height, width)
        out = tf.reshape(out, (-1, 224, 224, 3)) # make generic input shape
        return out


class MotionBlur:
    def __init__(self, size=3, angle=0):
        self.size = size
        self.kernel = self.create_kernel(size, angle)

    @staticmethod
    def create_kernel(size, angle):
        num_channels = 3
        kernel = np.zeros((size, size, num_channels, 1))
        kernel[:, np.shape(kernel)[1] // 2] = 1.
        kernel = rotate(kernel, angle, reshape=False, order=0, mode='wrap')
        kernel /= (np.sum(kernel)/num_channels)
        kernel = tf.convert_to_tensor(kernel, dtype=tf.float32)
        return kernel

    def __call__(self, img):
        img = tf.nn.depthwise_conv2d(img, self.kernel, strides=[1, 1, 1, 1], padding="SAME")
        return img


class RandomMotion:
    def __init__(self, min_amt=1, max_amt=32):
        self.num_channels = 3
        self.min_amt = min_amt
        self.max_amt = max_amt
        self.kernels = self.precompute_kernels(min_amt, max_amt)

    def precompute_kernels(self, min_amt=1, max_amt=32):
        kernels = []
        for size in range(min_amt, max_amt):
            kernel = self.create_kernel(size=size)
            kernel = utilities.pad_to_width(kernel, width=self.max_amt)
            kernels.append(kernel)
        return kernels

    def create_kernel(self, size):
        kernel = np.zeros((size, size, self.num_channels, 1))
        kernel[:, np.shape(kernel)[1] // 2] = 1.
        kernel /= (np.sum(kernel)/self.num_channels)
        return kernel

    def select_random_kernels(self, batch_size):
        idx = np.random.randint(1, len(self.kernels), size=batch_size)
        kernel = [self.kernels[i] for i in idx]
        return kernel, idx

    def reshape_kernel(self, kernel):
        batch_size, fh, fw, channels, out_channels = tf.shape(kernel)
        kernel = tf.transpose(kernel, [1, 2, 0, 3, 4])
        kernel = tf.reshape(kernel, [fh, fw, channels * batch_size, out_channels])
        return kernel

    def reshape_input(self, x):
        batch_size, height, width, channels = tf.shape(x)
        inp_r = tf.transpose(x, [1, 2, 0, 3])
        inp_r = tf.reshape(inp_r, [1, height, width, batch_size * channels])
        return inp_r

    def reshape_output(self, out, batch_size, height, width):
        channels = 3
        out = tf.reshape(out, [height, width, batch_size, channels])
        out = tf.transpose(out, [2, 0, 1, 3])
        return out

    def select_random_rotations(self, kernels):
        angles = np.random.randint(0, 91, size=len(kernels))
        return angles

    def rotate_kernels(self, kernels, angles):
        for idx in range(len(kernels)):
            kernels[idx] = rotate(kernels[idx], angles[idx], reshape=False,
                                 order=0, mode='wrap')
        return kernels

    def __call__(self, x):
        batch_size, height, width, channels = tf.shape(x)
        kernels, amounts = self.select_random_kernels(batch_size)
        angles = self.select_random_rotations(kernels)
        kernels = self.rotate_kernels(kernels, angles)
        kernels = self.reshape_kernel(kernels)
        x = self.reshape_input(x)
        kernels = tf.cast(kernels, tf.float32)
        out = tf.nn.depthwise_conv2d(x, filter=kernels, strides=[1, 1, 1, 1], padding='SAME')
        out = self.reshape_output(out, batch_size, height, width)

        return out, amounts



