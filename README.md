<p align="center">
  <img src="assets/tensorblur.png?raw=true" alt="TensorBlur" width=50%;/>
</p>

# TensorBlur: Efficient Image Blurring Routines in TensorFlow

# Contents
1. [Description](#Description)
2. [Quickstart](#Quickstart)
3. [Sources](#Sources)

## Description
This package provides methods for efficient image blurring using TensorFlow. 

These methods can be readily used in two ways:

1) A layer in a TensorFlow graph (i.e. a neural network), 
2) A standalone processing function

TensorBlur takes advantage of several convolutional tricks and GPU acceleration to make these methods extremely efficient.

## Quick Start
Apply blurring to a single image:
```python
import numpy as np
from PIL import Image
from tensorblur.gaussian import GaussianBlur

# Load an image
img = np.array(Image.open("assets/example2.jpg"))
# Create blur object
blur = GaussianBlur(size=7)
# Apply blurring
result = blur.apply(img)
```

Create a blur layer in a neural network
```python
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorblur import BlurLayer

# Load an image
img = np.array(Image.open("assets/example2.jpg"))

# Create Model with blur layer
inputs = tf.keras.layers.Input(shape=(128, 128, 3))
outputs = BlurLayer(min_amt=13, max_amt=13)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# convert input to tensor
img = tf.convert_to_tensor([img], tf.float32)     

# Apply model (call `model()`)
result = model(img)
```


## Sources
https://stackoverflow.com/questions/52012657/how-to-make-a-2d-gaussian-filter-in-tensorflow

https://computergraphics.stackexchange.com/questions/39/how-is-gaussian-blur-implemented

http://rastergrid.com/blog/2010/09/efficient-gaussian-blur-with-linear-sampling/

https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728

