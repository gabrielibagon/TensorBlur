import tensorflow as tf


class Blur:
    """
    Base class for Blur objects.
    """

    def __init__(self, size=1):
        """
        Initialize a Blur object with a specified kernel size.
        https://en.wikipedia.org/wiki/Kernel_(image_processing)

        Note: blurring is most efficient when size is odd.

        Args:
            size: pixel height and width of a square kernel
        """
        self.size = max(size, 1)
        self.kernel = self.create_kernel(size=self.size)

    def create_kernel(self, size=1):
        """
        Create kernel to apply blurring. Must be implemented by child class

        Args:
            size: pixel height and width of a square kernel

        Returns:
            kernel of shape [filter_height, filter_width, in_channels, channel_multiplier]
        """
        raise NotImplementedError

    def apply(self, img):
        """
        Apply blurring to an image or list of images

        Args:
            img: an image or list of images. Format is [height, width, channels]
             or [number of images, height, width, channels]. Note: all images must be
             of the same dimension.

        Returns:
            processed image of the same dimensions as the input
        """
        img_dim = img.shape
        img = self.format_input(img)
        img = tf.nn.depthwise_conv2d(img, self.kernel, strides=[1, 1, 1, 1], padding="SAME")
        img = self.format_output(img, img_dim)
        return img

    @staticmethod
    def format_input(img):
        """
        Format an input prior to blurring. Single images are reshaped to
        contain a batch dimension. Output is shape: [number of images, height, width, channels]

        Args:
            img: image or list of images

        Returns:
            input image(s) with shape = [number of images, height, width, channels]
        """
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
        """
        Format the output after blurring to have the same shape as the intial input.

        Args:
            img: output image or list of images
            img_dim: initial dimensions of image

        Returns:
            reshaped output image(s)
        """
        img = tf.reshape(img, img_dim)
        return img




