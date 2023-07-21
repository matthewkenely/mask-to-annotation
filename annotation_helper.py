# Dependencies required for segment-anything
# %pip install git+https://github.com/facebookresearch/segment-anything.git
# %pip install torch torchvision
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import cv2
import numpy as np

# Segment-anything model
# Downloading model checkpoint:
# Utilise the following link: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# Place the downloaded checkpoint in a new directory named Sam_checkpoints
sam_checkpoint = "Sam_checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mask_generator = SamAutomaticMaskGenerator(sam)


def single_object_bounding_box(mask):
    # transforming image into a binary image
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # outlining the contours in the image
    contours, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # variables to store the minimum and maximum x, y coordinates
    min_x = min_y = float('inf')
    max_x = max_y = 0

    # looping through all the contours and finding the minimum and maximum x, y coordinates
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    # calculating the width and height of the bounding box
    bounding_box_width = max_x - min_x
    bounding_box_height = max_y - min_y

    # creating the single bounding box using the calculated coordinates
    single_bounding_box = (
        min_x, min_y, bounding_box_width, bounding_box_height)

    return [single_bounding_box]


def polygon_approximation(mask, epsilon):
    # transforming image into a binary image
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # increasing standard deviation to blur more (repairing the mask)
    mask = cv2.GaussianBlur(mask, (7, 7), sigmaX=1, sigmaY=1)

    # Applying dilation and erosion to the mask
    kernel = np.ones((3, 3), np.uint8)
    # dilated_mask = cv2.dilate(mask, dilation_kernel, iterations=1)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    # outlining the contours in the image
    contours, _ = cv2.findContours(
        eroded_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Looping through the contours
    sorted_contours = []
    for contour in contours:
        # area = cv2.contourArea(contour)
        # if area > 2000:  # Example area threshold
        #     # Approximating the polygon to reduce the number of points
        #     # Adjust the epsilon value as needed
        # epsilon = epsilon * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(
            contour, epsilon * cv2.arcLength(contour, True), True)
        sorted_contours.append(approx_contour)

    # Sorting the contours based on the y coordinate of the bounding box
    sorted_contours = sorted(
        sorted_contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    return sorted_contours


def k_means_clustering(mask, epsilon, num_clusters):
    # transforming image into a binary image
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # increasing standard deviation to blur more (repairing the mask)
    mask = cv2.GaussianBlur(mask, (7, 7), sigmaX=1, sigmaY=1)

    # Applying dilation and erosion to the mask
    dilation_kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(mask, dilation_kernel, iterations=3)
    eroded_mask = cv2.erode(dilated_mask, dilation_kernel, iterations=1)

    # outlining the contours in the image
    contours, _ = cv2.findContours(
        eroded_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Looping through the contours
    sorted_contours = []
    for contour in contours:
        # area = cv2.contourArea(contour)
        # if area > 2000:  # Example area threshold
        #     # Approximating the polygon to reduce the number of points
        #     # Adjust the epsilon value as needed
        # epsilon = epsilon * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(
            contour, epsilon * cv2.arcLength(contour, True), True)
        sorted_contours.append(approx_contour)

    # Flatten the contours and convert to np.float32
    flattened_points = np.concatenate(
        sorted_contours).squeeze().astype(np.float32)

    # Using k-means clustering to find cluster centers
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        flattened_points, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Converting back to contour format with int32 data type
    kmeans_contours = [center.reshape(
        (-1, 1, 2)).astype(np.int32) for center in centers]

    # Creating a convex hull using all the cluster centers
    all_cluster_centers = np.concatenate(kmeans_contours)
    convex_hull = cv2.convexHull(all_cluster_centers)

    # Drawing the convex hull to form the polygon annotation
    annotations = [convex_hull]

    return annotations


def segment_anything(mask):
    # generating masks using segment-anything
    masks = mask_generator.generate(mask)
    print("Number of masks generated: ", len(masks))
    # transforming image into a binary image
    _, binary_mask = cv2.threshold(masks, 127, 255, cv2.THRESH_BINARY)

    # finding contours in the binary mask
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours
