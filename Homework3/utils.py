from skimage import io
from skimage.util import img_as_float
from skimage.color import rgb2gray
import numpy as np
from scipy.ndimage import correlate
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def computeHistogram(img_file, F, textons):
    """
    Compute the Bag of Words histogram representation of an image.

    Parameters:
    img_file (str): The file path of the image.
    F (ndarray): The filter bank array, where each filter is used for texture analysis.
    textons (ndarray): The array of K cluster centers, each being a 48D vector.

    Returns:
    ndarray: A histogram representing the frequency of each 'visual word' in the image.
    """

    # Load the image and convert to grayscale if it has three channels (RGB)
    image = io.imread(img_file)
    if len(image.shape) == 3:
        image = rgb2gray(image)
    # Convert the image to floating point representation for filter application
    image = img_as_float(image)

    # Get the dimensions of the image and the number of filters
    height, width = image.shape
    nf = F.shape[2]

    # Prepare an array to store the filter response of each pixel
    pixel_vectors = np.zeros((height, width, nf))

    # Apply each filter in the filter bank to the image
    for i in range(nf):
        filter = F[:, :, i]
        filtered_image = correlate(image, filter, mode='nearest')
        pixel_vectors[:, :, i] = filtered_image

    # Reshape the 3D array to 2D where each row represents a pixel's filter responses
    flat_pixel_vectors = pixel_vectors.reshape(-1, nf)

    # Compute the distance of each pixel's filter response to the cluster centers (textons)
    distances = cdist(flat_pixel_vectors, textons)
    nearest_textons = np.argmin(distances, axis=1)

    # Create a histogram representing the frequency of each texton in the image
    K = textons.shape[0]
    bow_hist = np.histogram(nearest_textons, bins=np.arange(K + 1), density=True)[0]

    return bow_hist


def createTextons(F, file_list, K):
    """
    Create textons from training images using a filter bank and K-means clustering.

    Parameters:
    F (ndarray): The filter bank with different filters.
    file_list (list of str): List of filenames of training images.
    K (int): The number of clusters to form (number of textons).

    Returns:
    ndarray: The array of textons, each being a 48D cluster center.
    """

    nf = F.shape[2]  # Total number of filters

    # List to store pixel response vectors from all images
    images_pixel_vectors = []

    # Process each image in the file list
    for img_name in file_list:
        # Load the image and convert to grayscale if necessary
        image = io.imread(img_name)
        if len(image.shape) == 3:
            image = rgb2gray(image)
        image = img_as_float(image)

        # Prepare to store the filter responses of this image
        height, width = image.shape
        pixel_vectors = np.zeros((height, width, nf))

        # Apply the filters to the image
        for i in range(nf):
            filter = F[:, :, i]
            filtered_image = correlate(image, filter, mode='nearest')
            pixel_vectors[:, :, i] = filtered_image

        # Store the pixel vectors
        images_pixel_vectors.append(pixel_vectors)

    # Choose a subset of pixels from each image to avoid high computational load
    num_samples_per_image = 100
    all_sampled_vectors = []

    for pixel_vectors in images_pixel_vectors:
        # Flatten the pixel response vectors to a 2D array for sampling
        flat_pixel_vectors = pixel_vectors.reshape(-1, nf)

        # Randomly sample a set number of pixels from the image
        indices = np.random.choice(flat_pixel_vectors.shape[0], num_samples_per_image, replace=False)
        sampled_vectors = flat_pixel_vectors[indices]

        # Collect the sampled vectors from all images
        all_sampled_vectors.extend(sampled_vectors)

    # Use K-means clustering to identify textons
    kmeans = KMeans(n_clusters=K, random_state=0)
    kmeans.fit(all_sampled_vectors)

    # Return the centers of the clusters (textons)
    return kmeans.cluster_centers_
