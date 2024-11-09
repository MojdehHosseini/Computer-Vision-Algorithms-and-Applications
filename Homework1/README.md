# Harris Corner Detector

The assignments of Compuer Vision Course in University

The Harris corner detector involves the following steps:# Harris Corner Detector

## Overview
This script implements the Harris Corner Detector, a feature detection algorithm used in computer vision to identify corners in images.

## Dependencies
- Python 3
- NumPy
- Matplotlib
- skimage

## Algorithm Steps
1. **Gradient Calculation**:
   - Calculate the image gradients in the x and y directions using Sobel filters.
2. **Gradient Products**:
- **Purpose**: Gradients in the x and y directions represent the rate of change in pixel intensities.
- **Edge Identification**: These gradients are fundamental in identifying edges in an image, as significant changes in intensity often correspond to edges.

## Gradient Products for Corner Detection
- **Beyond Edges**: While gradients help in edge detection, corner detection requires identifying locations where edges meet or intersect.
- **Dual-Dimension Analysis**: To detect corners, the algorithm analyzes changes in intensity across both x and y dimensions, necessitating the computation of gradient products.

## Gradient Products - Mathematical Explanation
- **Key Products**:
  - \( I_x^2 \): Square of the gradient in the x-direction, highlighting areas with high horizontal changes.
  - \( I_y^2 \): Square of the gradient in the y-direction, highlighting areas with high vertical changes.
  - \( I_{xy} \): Product of the gradients in both x and y directions, indicating the variation in intensity in both directions.
- **Corner Identification**: Analyzing \( I_x^2 \), \( I_y^2 \), and \( I_{xy} \) together allows the algorithm to distinguish corners from mere edges.

## Intuition Behind Gradient Products
- **Corner Characteristics**: Corners are characterized by significant variations in gradients in multiple directions.
- **Combined Behavior**: By evaluating the combined behavior of \( I_x^2 \), \( I_y^2 \), and \( I_{xy} \), the algorithm can effectively discern corners, as they will exhibit high values in both horizontal and vertical gradient changes.

## Forming the Structure Tensor
- **Role**: The gradient products are used to form a structure tensor (or second-moment matrix) for each pixel.
- **Harris Response**: The Harris corner response is then calculated based on this tensor, determining whether a pixel is part of a corner.

   - Compute the products of these gradients: \( I_x^2 \), \( I_y^2 \), and \( I_{xy} = I_x \times I_y \).
3. **Gaussian Filtering**:
   - Apply a Gaussian filter to smooth these gradient products, resulting in \( S_{xx} \), \( S_{yy} \), and \( S_{xy} \).
4. **Corner Response Computation**:
   - Calculate the Harris corner response \( R \) for each pixel using the formula:
     \[ R = \text{det}(M) - k \times (\text{trace}(M))^2 \]
     where \( M = \begin{bmatrix} S_{xx} & S_{xy} \\ S_{xy} & S_{yy} \end{bmatrix} \), \( \text{det}(M) = S_{xx} \times S_{yy} - S_{xy}^2 \), and \( \text{trace}(M) = S_{xx} + S_{yy} \). The parameter \( k \) is typically around 0.04 to 0.06.
5. **Thresholding**:
   - Apply a threshold to the response map to filter out weak corner responses.
6. **Non-Maximum Suppression**:
   - Perform non-maximum suppression using `peak_local_max` from `skimage` to find local maxima in the corner response map.

## Running the Script
To run this script, use the following command:



## Output
The script generates three figures:
1. The corner response map after computing the response.
2. The thresholded response map.
3. The original image with detected corners marked.

## Customization
You can adjust parameters such as the window size, sensitivity factor `k`, and threshold value for different results.
