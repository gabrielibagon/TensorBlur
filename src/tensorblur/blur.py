import tensorflow as tf


class Blur:
    """Generic class for Blur objects."""

    def __init__(self, size=None):
        self.size = max(size, 1)
        self.kernel = self.create_kernel()

    def create_kernel(self):
        """Create kernel to apply blurring"""
        raise NotImplementedError

    def blur(self, img):
        """Function to apply blurring to an image via depthwise convolution"""

        if len(tf.shape(img)) == 3:
            img = tf.expand_dims(img, 0)
        elif tf.shape(img) != 4:
            num_dims = len(tf.shape(img))
            img_shape = tf.shape(img).numpy()
            raise ValueError('Input image must have shape '
                             '[ batch height width channels ] or [ height width channels ]. '
                             f'Current input has {num_dims} dimensions: {img_shape}')

        img = tf.cast(img, tf.float32)
        img = tf.nn.depthwise_conv2d(img, self.kernel, strides=[1, 1, 1, 1], padding="SAME")
        return img
