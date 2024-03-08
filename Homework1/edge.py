import numpy as np
import matplotlib.pylab as plt
from skimage import io
from utils import gaussian_kernel, filter2d, partial_x, partial_y

def main():
    # Load image
    img = io.imread('iguana.png', as_gray=True)

    ### YOUR CODE HERE

    # Smooth image with Gaussian kernel
    kernel = gaussian_kernel(5, 1)
    img_smoothed = filter2d(img, kernel)

    # Compute x and y derivative on smoothed image
    img_x = partial_x(img_smoothed)
    img_y = partial_y(img_smoothed)

    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(img_x ** 2 + img_y ** 2)

    # Visualize results
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(24, 12))  # Increase the figure size
    plt.subplot(1, 3, 1)
    plt.title('Gradient in x direction')
    plt.imshow(img_x, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('Gradient in y direction')
    plt.imshow(img_y, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('Gradient Magnitude')
    plt.imshow(gradient_magnitude, cmap='gray')

    plt.savefig('Edge_Detector.jpg')
    plt.show()


    ### END YOUR CODE
    
if __name__ == "__main__":
    main()

