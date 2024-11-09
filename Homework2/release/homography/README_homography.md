
# Homography and Image Matching Project

This project demonstrates key concepts in computer vision, including homography computation, feature matching, and image compositing. It uses SIFT features to match keypoints between images and applies RANSAC to compute a robust homography. The final result overlays one image onto another using the computed homography.

---

## Features

- **SIFT Feature Matching**: Detect and match keypoints between two images.
- **RANSAC for Homography**: Estimate a robust homography matrix with inlier matches.
- **Bounding Box Visualization**: Visualize the detected region in the target image.
- **Image Compositing**: Overlay one image onto another using the computed homography.

---

## Project Structure

```
.
├── homography.py      # Core implementation for homography computation and matching.
├── run.py             # Main script to run the entire pipeline.
├── utils.py           # Utility functions for visualization.
├── cv_cover.jpg       # Cover image of the book "Computer Vision".
├── cv_desk.jpg        # Photo of the book on a desk.
├── hp_cover.jpg       # Cover image of "Harry Potter".
└── final result.jpg   # Composite image (generated after running the pipeline).
```

---

## Getting Started

### Prerequisites

Ensure you have Python installed along with the following packages:

- `numpy`
- `opencv-python`
- `matplotlib`
- `scikit-image`

You can install these dependencies with:

```bash
pip install numpy opencv-python matplotlib scikit-image
```

---

### Usage

1. **Prepare Your Environment**:
   - Place the input images (`cv_cover.jpg`, `cv_desk.jpg`, `hp_cover.jpg`) in the same directory as the scripts.

2. **Run the Pipeline**:
   - Execute the main script:

     ```bash
     python run.py
     ```

3. **Output**:
   - The script generates intermediate visualizations:
     - Raw matches between keypoints.
     - Matches after RANSAC filtering.
     - Bounding box visualization.
   - A final composite image (`final result.jpg`) that overlays `hp_cover.jpg` onto `cv_desk.jpg`.

---

## How It Works

1. **Keypoint Matching**:
   - Uses SIFT to extract keypoints and descriptors from images.
   - Matches keypoints using a Brute-Force matcher with a ratio test.

2. **Homography Estimation**:
   - Uses RANSAC to estimate a robust homography matrix.
   - Filters out outlier matches to ensure accuracy.

3. **Image Compositing**:
   - Warps the "Harry Potter" cover (`hp_cover.jpg`) to align with the "Computer Vision" book's bounding box on the desk (`cv_desk.jpg`).

4. **Visualization**:
   - Displays intermediate and final results for better understanding.

---

## Example Workflow

1. **Input Images**:
   - `cv_cover.jpg`: Cover image of the "Computer Vision" book.
   - `cv_desk.jpg`: Photo of the book placed on a desk.
   - `hp_cover.jpg`: Cover image of "Harry Potter".

2. **Intermediate Steps**:
   - Match keypoints between `cv_cover.jpg` and `cv_desk.jpg`.
   - Compute homography and visualize the bounding box.
   - Overlay `hp_cover.jpg` onto the detected region.

3. **Output**:
   - `final result.jpg`: Composite image showing "Harry Potter" overlaid on the desk.

---

## Notes

- Ensure all input images are in the correct format and dimensions.
- You can modify thresholds and parameters in `homography.py` to tune the results.
- Resize or preprocess the images for better performance if needed.

---

## License

This project is open-source and available under the MIT License.
