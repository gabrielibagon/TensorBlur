import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorblur.gaussian import GaussianBlur

# Load an image
init_img = np.array(Image.open("assets/example1.jpg"))

# Iterate over different blur sizes
for size in [1, 5, 9, 17, 33, 65]:

    # Make a fresh copy of the image
    img = init_img.copy()

    # Create apply object
    gauss = GaussianBlur(size=size)

    # Apply blurring
    result = gauss.apply(img)
    result = result.numpy().astype(int)

    # Display result
    plt.title('Gaussian Blur | Size: %i' % size)
    plt.imshow(result)
    plt.axis('off')
    plt.show()
