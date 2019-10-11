import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from tensorblur.gaussian import GaussianBlur

# Load an image
init_img = np.array(Image.open("assets/example2.jpg"))

for size in [2, 4, 8, 16, 32, 64]:

    # Make a fresh copy of the image
    img = init_img.copy()

    # Create apply object
    gauss = GaussianBlur(size=size)

    # Apply blurring
    img = gauss.apply(img)

    # Display result
    plt.title('Gaussian Blur | Size: %i' % size)
    plt.imshow(img[0, ...].numpy().astype(int))
    plt.show()


    # Apply blurring
    img = gauss.apply(img)

    # Display result
    plt.title('Gaussian Blur | Size: %i' % size)
    plt.imshow(img[0, ...].numpy().astype(int))
    plt.show()


    # Apply blurring
    img = gauss.apply(img)

    # Display result
    plt.title('Gaussian Blur | Size: %i' % size)
    plt.imshow(img[0, ...].numpy().astype(int))
    plt.show()
