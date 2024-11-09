
# Texture Classification with LM Filter Bank and Bag-of-Words

This repository implements a texture classification pipeline using the Leung-Malik (LM) Filter Bank and Bag-of-Words (BoW) representation. It includes scripts for training a model on texture images and testing new images against the trained model.

## Features

- Implements the **Leung-Malik Filter Bank** for texture analysis.
- Creates a Bag-of-Words representation of textures using K-means clustering.
- Trains a texture classification model by computing histograms of visual words.
- Matches test images to training images using histogram comparisons.

---

## Project Structure

```
.
├── utils.py           # Utility functions for texton creation and histogram computation.
├── LMFilters.py       # Implementation of the Leung-Malik Filter Bank.
├── run_train.py       # Script for training the texture classification model.
├── run_test.py        # Script for testing the trained model on new images.
├── train1.jpg, ...    # Training images (replace with your dataset).
├── test1.jpg, ...     # Test images (replace with your dataset).
└── model.pkl          # Saved model file (generated after training).
```

---

## Getting Started

### Prerequisites

Ensure you have Python installed along with the following packages:

- `numpy`
- `scikit-image`
- `scipy`
- `scikit-learn`
- `opencv-python`
- `matplotlib`

You can install these dependencies with:

```bash
pip install numpy scikit-image scipy scikit-learn opencv-python matplotlib
```

---

### Usage

#### Training

1. Place your training images in the working directory, named as `train1.jpg`, `train2.jpg`, ..., `trainN.jpg`.
2. Run the training script:

   ```bash
   python run_train.py
   ```

   This will:
   - Generate textons (clusters of filter responses) using the training images.
   - Compute histograms of visual words for the training images.
   - Save the model to `model.pkl`.

---

#### Testing

1. Place your test images in the working directory, named as `test1.jpg`, `test2.jpg`, ..., `testM.jpg`.
2. Run the testing script:

   ```bash
   python run_test.py
   ```

   This will:
   - Load the trained model from `model.pkl`.
   - Compute histograms for the test images.
   - Predict the closest training image for each test image.

---

## How It Works

1. **Filter Bank Creation**: The LM Filter Bank consists of various filters (e.g., Gaussian, Laplacian of Gaussian) applied at multiple orientations and scales.
2. **Texton Creation**: Filter responses from training images are clustered using K-means to form textons, which represent texture features.
3. **Histogram Computation**: Each image is represented as a histogram of textons (visual words).
4. **Classification**: Test images are compared to training images using histogram distances to find the closest match.

---

## Example Workflow

1. Prepare your training images:
   ```
   train1.jpg, train2.jpg, ..., trainN.jpg
   ```

2. Train the model:
   ```bash
   python run_train.py
   ```

3. Prepare your test images:
   ```
   test1.jpg, test2.jpg, ..., testM.jpg
   ```

4. Test the model:
   ```bash
   python run_test.py
   ```

5. Example output:
   ```
   For test1.jpg, the closest training image is train3.jpg
   For test2.jpg, the closest training image is train1.jpg
   ```

---

## Notes

- Adjust the number of clusters (`K`) in `run_train.py` for optimal performance.
- Ensure consistent image formats (e.g., grayscale or RGB) across training and testing datasets.

---

## References

- T. Leung and J. Malik. "Representing and recognizing the visual appearance of materials using three-dimensional textons." *International Journal of Computer Vision*, 43(1):29-44, June 2001.
- LM Filter Bank Reference: [Oxford Robots](http://www.robots.ox.ac.uk/~vgg/research/texclass/filters.html)

---

## License

This project is open-source and available under the MIT License. See the `LICENSE` file for details.
