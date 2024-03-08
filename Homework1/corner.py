import numpy as np
from utils import filter2d, partial_x, partial_y,gaussian_kernel
from skimage.feature import peak_local_max
from skimage.io import imread
import matplotlib.pyplot as plt

def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    response = None
    
    ### YOUR CODE HERE

    # Compute gradients
    Ix = partial_x(img)
    Iy = partial_y(img)

    # Compute products of derivatives
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    # Apply Gaussian filter
    gauss = gaussian_kernel(window_size, window_size / 2)
    Sxx = filter2d(Ixx, gauss)
    Syy = filter2d(Iyy, gauss)
    Sxy = filter2d(Ixy, gauss)

    # Compute corner response
    detM = Sxx * Syy - Sxy ** 2
    traceM = Sxx + Syy
    response = detM - k * traceM ** 2

    ### END YOUR CODE

    return response

def main():

    img = imread('building.jpg', as_gray=True)

    ### YOUR CODE HERE
    

    # Compute Harris corner response
    response = harris_corners(img)

    # Threshold on response
    thresh = 0.01 * np.max(response)
    response_thresh = response > thresh

    # Perform non-max suppression by finding peak local maximum
    corners = peak_local_max(response, min_distance=1, threshold_abs=thresh)

    # Visualize results
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(24, 12))  # Increase the figure size
    plt.subplot(1, 3, 1)
    plt.title('Harris Response')
    plt.imshow(response, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('Thresholded Response')
    plt.imshow(response_thresh, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('Detected Corners')
    plt.imshow(img, cmap='gray')
    for corner in corners:
        plt.scatter(corner[1], corner[0], s=40, c='red', marker='x')  # Increase the scatter point size

    plt.savefig('Corner_Detector.jpg')
    plt.show()


    
if __name__ == "__main__":
    main()
