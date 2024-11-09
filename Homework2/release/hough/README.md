
# Lane Detection Using Hough Transform

This project demonstrates lane detection on a road image using the Hough Transform. The process includes edge detection, masking a region of interest (ROI), and identifying lanes by finding peaks in the Hough accumulator space.

---

## Features

- **Canny Edge Detection**: Detect edges in the road image.
- **Region of Interest Masking**: Focus on the area of the image where lanes are most likely to appear.
- **Hough Transform**: Detect lane lines by finding peaks in the Hough accumulator.
- **Lane Visualization**: Overlay detected lanes on the original image.

---

## Project Structure

```
.
├── hough.py                 # Main script implementing the Hough Transform for lane detection.
├── utils.py                 # Helper functions for creating masks and transforming lines.
├── road.jpg                 # Input road image for lane detection.
├── Blue and Orange Lines.jpg # Output image with detected lanes.
├── Edges iin ROI.jpg        # Edges in the region of interest.
├── Mask.jpg                 # Region of interest mask.
├── Edges.jpg                # Detected edges using Canny.
├── canny_edges.jpg          # Additional visualization of Canny edges.
├── masked_edges.npy         # Numpy array of masked edges.
```

---

## Getting Started

### Prerequisites

Ensure you have Python installed along with the following packages:

- `numpy`
- `matplotlib`
- `opencv-python`
- `scikit-image`

You can install these dependencies with:

```bash
pip install numpy matplotlib opencv-python scikit-image
```

---

### Usage

1. **Prepare the Environment**:
   - Place the `road.jpg` image in the same directory as the scripts.

2. **Run the Script**:
   - Execute the main script:

     ```bash
     python hough.py
     ```

3. **Output**:
   - The script generates the following visualizations:
     - Detected edges (`Edges.jpg`).
     - Mask (`Mask.jpg`).
     - Edges within the ROI (`Edges iin ROI.jpg`).
     - Final output with detected lanes (`Blue and Orange Lines.jpg`).

---

## How It Works

1. **Edge Detection**:
   - Apply the Canny Edge Detector to highlight edges in the input image.

2. **Region of Interest Masking**:
   - Focus only on the region where lanes are expected using a mask.

3. **Hough Transform**:
   - Map edge points to the Hough space to identify straight lines.
   - Find peaks in the accumulator space to determine lane lines.

4. **Line Visualization**:
   - Convert Hough space peaks into lane lines and overlay them on the input image.

---

## Example Workflow

1. **Input**:
   - `road.jpg`: Road image with lanes to detect.

2. **Intermediate Steps**:
   - Detect edges using Canny (`Edges.jpg`).
   - Mask region of interest (`Mask.jpg`).
   - Extract edges in ROI (`Edges iin ROI.jpg`).

3. **Output**:
   - Detected lanes (`Blue and Orange Lines.jpg`).

---

## Notes

- Adjust thresholds and parameters in `hough.py` for optimal performance.
- Use high-resolution images for better results.

---

## License

This project is open-source and available under the MIT License.
