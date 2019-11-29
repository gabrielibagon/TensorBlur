import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorblur.layer import BlurLayer

# Load an image
img = np.array(Image.open("assets/example1.jpg"))

# Create Model with blur layer
blur_amt = 50

inputs = tf.keras.layers.Input(shape=img.shape)
outputs = BlurLayer(size=blur_amt)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary()

# Prepare input
img = [img]                                     # add batch dimension
img = tf.convert_to_tensor(img, tf.float32)     # convert to tensor

# Apply model (call `model()`)
result = model(img)

# Format output
result = result.numpy()         # convert to numpy array
result = result[0]              # remove batch dimension
result = result.astype(int)     # convert to int (for display)

# Display output image
plt.title('Gaussian Blur | Size: %i' % blur_amt)
plt.imshow(result)
plt.axis('off')
plt.show()
