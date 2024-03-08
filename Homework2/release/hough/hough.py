# import other necessary libaries
import numpy as np
import matplotlib.pylab as plt
from skimage import io
import cv2
from utils import create_line, create_mask


def non_max_suppression(accumulator, x, y, radius=10):
    """ Suppress values of neighboring cells in the accumulator array. """
    height, width = accumulator.shape
    # Iteration over a square region centered at (x,y)
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            # Checking if the cell lies within the bounds of accumulator array.
            if (0 <= x + i < height) and (0 <= y + j < width):
                # Suppressing the cell
                accumulator[x + i, y + j] = 0

def main():

    # load the input image
    # image = io.imread('hough/road.jpg', as_gray=True)
    image = cv2.imread('road.jpg', cv2.IMREAD_GRAYSCALE)

    # run Canny edge detector to find edge points
    edges = cv2.Canny(image, 30, 120)


    plt.imshow(edges, cmap='gray')
    plt.title('Edges')
    plt.savefig('Edges.jpg')
    plt.show()

    # Get the dimensions of the image
    H, W = edges.shape

    # create a mask for ROI by calling create_mask
    mask = create_mask(H, W)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.savefig('Mask.jpg')
    plt.show()


    # extract edge points in ROI by multipling edge map with the mask
    masked_edges = edges * mask

    # After processing the image
    plt.imshow(masked_edges, cmap='gray')
    plt.title('Edges iin ROI')
    plt.savefig('Edges iin ROI.jpg')
    plt.show()




    # perform Hough transform

    # Define the Hough space
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = edges.shape
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some reusable values
    # The thetas array contains a range of angles in radians (as previously set up to range from -90 to 90 degrees)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    # This line simply stores the number of theta values (or angles) in the variable num_thetas
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho

    # Accumulator array that is used to keep track of potential lines in the Hough space
    # "2 * diag_len" is used as the number of rows, representing all possible values of rho (distance from origin to the line)
    # its maximum absolute value is the length of the image diagonal
    # "num_thetas" is the number of columns, representing all possible angles (thetas) a line can have.
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)

    # This line finds all the edge points in the Edges and returns the indices of nonzero elements in it
    y_idxs, x_idxs = np.nonzero(masked_edges)  # (row, col) indexes to edges

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            # This increment represents a vote for the existence of a line in the image space with parameters rho and theta
            accumulator[rho, t_idx] += 1


    # find the right lane by finding the peak in hough space
    idx = np.argmax(accumulator)
    rho = rhos[idx // accumulator.shape[1]]
    theta = thetas[idx % accumulator.shape[1]]
    line_blue = create_line(rho, theta,image)

    # zero out the values in accumulator around the neighborhood of the peak
    # Apply non-max-suppression around the found line
    non_max_suppression(accumulator, idx // accumulator.shape[1], idx % accumulator.shape[1],50)


    # find the left lane by finding the peak in hough space
    idx = np.argmax(accumulator)
    rho = rhos[idx // accumulator.shape[1]]
    theta = thetas[idx % accumulator.shape[1]]
    line_orange = create_line(rho, theta,image)

    # plot the results
    plt.imshow(image, cmap='gray')
    plt.plot(line_blue[0], line_blue[1], 'b')  # blue line
    plt.plot(line_orange[0], line_orange[1], 'orange')  # orange line
    plt.savefig('Blue and Orange Lines.jpg')
    plt.show()

if __name__ == "__main__":
    main()
