
# Project Summaries

This repository contains summaries of multiple projects related to computer vision and image processing. Below is a brief overview of each project:

---

## 1. **Texture Classification with LM Filter Bank and Bag-of-Words**
**Description**: 
- Implements the Leung-Malik (LM) Filter Bank for texture analysis.
- Uses a Bag-of-Words representation to classify textures in images.
**Main Features**:
- Texton generation using K-means clustering.
- Histogram-based texture classification.
- Scripts for training and testing a texture classification model.
**Key Files**:
- `utils.py`: Core utility functions for texton creation and histogram computation.
- `run_train.py`: Trains the model using texture images.
- `run_test.py`: Tests the model against test images.

---

## 2. **Homography and Image Matching**
**Description**:
- Demonstrates key concepts like feature matching, homography computation, and image compositing.
**Main Features**:
- SIFT feature detection and matching.
- RANSAC for robust homography estimation.
- Overlays one image onto another using computed homography.
**Key Files**:
- `homography.py`: Core implementation for homography and feature matching.
- `run.py`: Runs the entire pipeline.
- `utils.py`: Visualization utilities.

---

## 3. **Lane Detection Using Hough Transform**
**Description**:
- Detects lanes in road images using Canny Edge Detection and Hough Transform.
**Main Features**:
- Focuses on Region of Interest (ROI) using masking.
- Identifies lanes by finding peaks in the Hough accumulator space.
**Key Files**:
- `hough.py`: Implements Hough Transform for lane detection.
- `utils.py`: Functions for creating masks and line transformations.

---

## 4. **Harris Corner Detector**
**Description**:
- Implements the Harris Corner Detector to identify corners in images.
**Main Features**:
- Detects image corners using gradients and the Harris response formula.
- Includes thresholding and non-maximum suppression for corner localization.
**Key Files**:
- `harris_detector.py` (assumed script name): Implements the algorithm.

---

### Notes
Each project includes a more detailed README in its respective directory. The generated READMEs can be referenced for specific usage instructions, algorithm details, and outputs.

