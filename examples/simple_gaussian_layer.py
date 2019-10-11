import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorblur.layer import BlurLayer

# Load an image
init_img = np.array(Image.open("assets/example2.jpg"))

img2 = np.array(Image.open("assets/example1.png"))

size = 16
# Make a fresh copy of the image
img = init_img.copy()

# Create apply object
inputs = tf.keras.layers.Input(shape=(128, 128, 3))
outputs = BlurLayer(min_amt=4, max_amt=4)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()

img = tf.convert_to_tensor([img, img], tf.float32)
model(img)