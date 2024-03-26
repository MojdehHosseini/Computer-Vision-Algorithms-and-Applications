import cv2
import numpy as np

def compute_homography(src_pts, dst_pts):
    A = []
    for src, dst in zip(src_pts, dst_pts):
        x, y = src[0], src[1]
        x_prime, y_prime = dst[0], dst[1]

        # Form the equations and add to matrix A
        A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])

    A = np.array(A)

    # Apply SVD
    U, S, Vt = np.linalg.svd(A)

    # The homography is the last row of V (or last column of V^T)
    H = Vt[-1].reshape(3, 3)

    # Normalize the matrix
    H = H / H[-1, -1]

    return H

def matchPics(I1, I2):
    # Given two images I1 and I2, perform SIFT matching to find candidate match pairs

    ### YOUR CODE HERE
    ### You can use skimage or OpenCV to perform SIFT matching

    # Convert color images to grayscale, ensuring they are in the correct format for SIFT
    I1 = ((I1) * 255).astype('uint8')
    I2 = ((I2) * 255).astype('uint8')


    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect and compute SIFT features for image1
    locs1, descriptors_1 = sift.detectAndCompute(I1, None)

    # Detect and compute SIFT features for image2
    locs2, descriptors_2 = sift.detectAndCompute(I2, None)

    # Initialize Brute-Force matcher and use KNN approach to match
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    # Apply ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Process locs1 and locs2 to extract (x, y) coordinates from keypoint objects
    locs1 = np.array([kp.pt for kp in locs1]).astype(np.float32)
    locs2 = np.array([kp.pt for kp in locs2]).astype(np.float32)

    # Create a new matches array that corresponds to the format [(index1, index2), ...]
    matches = np.array([[m.queryIdx, m.trainIdx] for m in good_matches])

    return matches, locs1, locs2

def computeH_ransac(matches, locs1, locs2):
    # Placeholder for the best homography matrix and inliers
    bestH = None
    max_inliers = 0
    inliers = []

    # Number of iterations and inlier threshold
    num_iterations = 1000
    threshold = 2  # pixel distance

    for _ in range(num_iterations):
        # Step 1: Random Sampling
        chosen_matches = matches[np.random.choice(matches.shape[0], 4, replace=False)]
        selected_locs1 = locs1[chosen_matches[:, 0]]
        selected_locs2 = locs2[chosen_matches[:, 1]]


        # Step 2: Homography Estimation
        H = compute_homography(selected_locs1, selected_locs2)


        # Step 3: Inlier Counting
        num_inliers, current_inliers = 0, []
        for i, match in enumerate(matches):
            pt1 = np.append(locs1[match[0]], 1)
            pt2 = np.append(locs2[match[1]], 1)
            projected_pt1 = np.dot(H, pt1)
            projected_pt1 /= projected_pt1[2]  # Normalize

            if np.linalg.norm(projected_pt1[:2] - pt2[:2]) < threshold:
                num_inliers += 1
                current_inliers.append(i)
        # print(H, type(H))
        # Step 4: Best Homography Selection
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            bestH = H
            inliers = current_inliers

    print(bestH)
    return bestH, inliers


def compositeH(H, template, img):

    # Resize the template to the size of the target image
    # template_resized = cv2.resize(template, (img.shape[1], img.shape[0]))
    template_resized=template
    # Create a mask of the same size as the resized template
    mask = np.ones_like(template_resized) * 255  # Assuming template is grayscale

    # Warp the mask and the template using the homography
    warped_mask = cv2.warpPerspective(mask, H, (img.shape[1], img.shape[0]))
    warped_template = cv2.warpPerspective(template_resized, H, (img.shape[1], img.shape[0]))

    # Use the mask to combine the warped template and the original image
    composite_img = img.copy()
    composite_img[warped_mask > 0] = warped_template[warped_mask > 0]

    return composite_img
