import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from utils import gaussian_kernel, filter2d


def main():
    # load the image
    im = imread('paint.jpg').astype('float')
    im = im / 255

    # number of levels for downsampling
    N_levels = 5

    # make a copy of the original image for naive subsampling
    im_subsample_naive = im.copy()

    # naive subsampling, visualize the results on the 1st row
    for i in range(N_levels):
        # subsample image
        im_subsample_naive = im_subsample_naive[::2, ::2, :]
        plt.subplot(2, N_levels, i + 1)
        plt.imshow(im_subsample_naive)
        plt.axis('off')

    # make another copy of the original image for anti-aliasing subsampling
    im_subsample_aa = im.copy()

    # subsampling without aliasing, visualize results on 2nd row
    for i in range(N_levels):
        # Apply Gaussian filter before subsampling
        kernel = gaussian_kernel(l=5, sig=1.)
        im_subsample_aa = np.stack([filter2d(im_subsample_aa[:, :, c], kernel) for c in range(3)], axis=-1)
        im_subsample_aa = im_subsample_aa[::2, ::2, :]

        plt.subplot(2, N_levels, N_levels + i + 1)
        plt.imshow(im_subsample_aa)
        plt.axis('off')

    plt.savefig('Downsampled_Images.jpg')
    plt.show()



if __name__ == "__main__":
    main()
