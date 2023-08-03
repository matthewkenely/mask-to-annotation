import colorsys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Constants
# Noise Threshold
NOISE_THRESHOLD = 40


def single_object_bounding_box(mask, do_cvt):
    # transforming image into a binary image
    if do_cvt:
        # transforming image into a binary image
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # thresholding the image
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # increasing standard deviation to blur more (anti-aliasing)
    mask = cv2.GaussianBlur(mask, (7, 7), sigmaX=1, sigmaY=1)

    # applying dilation (optional) and erosion to the mask
    kernel = np.ones((3, 3), np.uint8)
    # dilated_mask = cv2.dilate(mask, dilation_kernel, iterations=1)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

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


def multiple_objects_bounding_box(mask, do_cvt):
    # retrieving the connected components
    components = component_labelling(mask)

    # list to store the bounding boxes
    bounding_boxes = []

    # iterating over all the connected components
    for label, component in components.items():
        contours, _ = cv2.findContours(
            component, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # sorting the contours based on their size, largest to smallest
        contours = sorted(contours, key=lambda ctr: cv2.contourArea(ctr))[::-1]

        # creating a bounding box for the largest connected component
        x, y, w, h = cv2.boundingRect(contours[0])
        bounding_box = (x, y, w, h)
        # appending the bounding box to the list
        bounding_boxes.append(bounding_box)
    return bounding_boxes


def single_object_polygon_approximation(mask, epsilon, do_cvt):
    # transforming image into a binary image
    if do_cvt:
        # transforming image into a binary image
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # thresholding the image
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # increasing standard deviation to blur more (anti-aliasing)
    mask = cv2.GaussianBlur(mask, (7, 7), sigmaX=1, sigmaY=1)

    # applying dilation (optional) and erosion to the mask
    kernel = np.ones((3, 3), np.uint8)
    # dilated_mask = cv2.dilate(mask, dilation_kernel, iterations=1)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    # outlining the contours in the image
    contours, _ = cv2.findContours(
        eroded_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # looping through the contours
    sorted_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > NOISE_THRESHOLD:  # removing small noise
            # Approximating the polygon to reduce the number of points
            approx_contour = cv2.approxPolyDP(
                contour, epsilon * cv2.arcLength(contour, True), True)
            sorted_contours.append(approx_contour)

    # sorting the contours based on the y coordinate of the bounding box
    sorted_contours = sorted(
        sorted_contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    return sorted_contours


def multiple_objects_polygon_approximation(mask, epsilon, do_cvt):
    # retrieving the connected components
    components = component_labelling(mask)
    # Polygon approximation
    object_contours = {}

    # iterating over all the labels from 1 to num_labels (inclusive)
    for target_label, component in components.items():
        # retrieving the contours of the selected connected component
        contours, _ = cv2.findContours(
            component, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # looping through the contours
        for contour in contours:
            # calculating the area of the contour
            area = cv2.contourArea(contour)
            if area > NOISE_THRESHOLD:  # removing small noise
                # approximating the polygon to reduce the number of points
                approx_contour = cv2.approxPolyDP(
                    contour, epsilon * cv2.arcLength(contour, True), True)

                # adding the contour to the dictionary
                if target_label not in object_contours:
                    object_contours[target_label] = []
                object_contours[target_label].append(approx_contour)

    return object_contours


def single_object_k_means_clustering(mask, max_clusters, do_cvt):
    # transforming image into a binary image
    if do_cvt:
        # transforming image into a binary image
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # thresholding the image
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # increasing standard deviation to blur more (repairing the mask)
    mask = cv2.GaussianBlur(mask, (7, 7), sigmaX=1, sigmaY=1)

    # applying dilation (optional) and erosion to the mask
    kernel = np.ones((3, 3), np.uint8)
    # dilated_mask = cv2.dilate(mask, dilation_kernel, iterations=1)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    # outlining the contours in the image
    contours, _ = cv2.findContours(
        eroded_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # sorting the contours based on the y coordinate of the bounding box
    sorted_contours = sorted(
        contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    # flattening the contours and convert to np.float32
    flattened_points = np.concatenate(
        sorted_contours).squeeze().astype(np.float32)

    # using k-means clustering to find cluster centers
    if max_clusters > len(flattened_points):
        max_clusters = len(flattened_points)

    # using the elbow method to find the optimal number of clusters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    '''
    cv2.kmeans
    ----------
    flattened_points :: samples
    max_clusters :: max_clusters
    bestLabels :: None
    criteria ::
        TERM_CRITERIA_EPS -> stop the algorithm iteration if specified accuracy, epsilon, is reached
        0.2 -> epsilon

        TERM_CRITERIA_MAX_ITER -> stop the algorithm after the specified number of iterations, max_iter
        100 -> max_iter 
    
    attempts :: 10 (using different initial labellings)
    flags :: KMEANS_RANDOM_CENTERS -> select random initial centers in each attempt
    '''

    _, labels, centers = cv2.kmeans(
        flattened_points, max_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # converting back to contour format with int32 data type
    kmeans_contours = [center.reshape(
        (-1, 1, 2)).astype(np.int32) for center in centers]

    # creating a convex hull using all the cluster centers
    all_cluster_centers = np.concatenate(kmeans_contours)
    convex_hull = cv2.convexHull(all_cluster_centers)

    # drawing the convex hull to form the polygon annotation
    annotations = [convex_hull]

    return annotations


def multiple_objects_k_means_clustering(mask, max_clusters, do_cvt):
    # retrieving the connected components
    components = component_labelling(mask)
    # K-means clustering
    # initializing a dictionary to store the contours for each connected component
    annotations = {}

    # iterating over all the labels from 1 to num_labels (inclusive)
    for label, component in components.items():
        # retrieving the contours of the selected connected component
        contours, _ = cv2.findContours(
            component, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # checking if the list of contours is empty
        if not contours:
            continue

        # flattening the contours and convert to np.float32
        flattened_points = np.concatenate(
            contours).squeeze().astype(np.float32)

        # using k-means clustering to find cluster centers
        if max_clusters > len(flattened_points):
            max_clusters = len(flattened_points)

        # using the elbow method to find the optimal number of clusters
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        _, labels, centers = cv2.kmeans(
            flattened_points, max_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # converting back to contour format with int32 data type
        kmeans_contours = [center.reshape(
            (-1, 1, 2)).astype(np.int32) for center in centers]

        # creating a convex hull using all the cluster centers
        all_cluster_centers = np.concatenate(kmeans_contours)
        convex_hull = cv2.convexHull(all_cluster_centers)

        # storing the contours in the dictionary with the label as the key
        annotations[label] = [convex_hull]

    return annotations


def component_labelling(image, dynamic_threshold_factor=0.0003):
    # dynamic threshold factor is used to calculate the dynamic threshold
    # checking if the input image is colored (3 channels) or binary (1 channel)
    if image.ndim == 3 and image.shape[-1] == 3:  # colored mask
        # converting the colored mask to HSV color space
        hsv_mask = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # retrieving the unique colors present in the image (excluding black and white)
        unique_colors, color_counts = np.unique(
            hsv_mask.reshape(-1, hsv_mask.shape[2]), axis=0, return_counts=True)

        # creating a mask for the background color
        background_mask = np.zeros(hsv_mask.shape[:2], dtype=np.uint8)
        background_mask[color_counts.argmin()] = 255

        # calculating the dynamic threshold based on the total number of pixels in the image
        min_pixel_threshold = int(
            dynamic_threshold_factor * np.prod(hsv_mask.shape[:2]))

        # creating a dictionary to store the masks for each color
        components = {}

        # defining a mask for each color and find contours for each mask
        for label, color in enumerate(unique_colors):
            # checking if the number of pixels for this color is greater than the threshold and not the background
            if color_counts[label] > min_pixel_threshold and not np.all(color == hsv_mask[background_mask == 255][0]):
                # Dynamic object identification using color-based segmentation
                lower_color = np.array(
                    [color[0] - 10, max(0, color[1] - 40), max(0, color[2] - 40)])
                upper_color = np.array(
                    [color[0] + 10, min(255, color[1] + 40), min(255, color[2] + 40)])

                # creating a mask for the selected color
                color_mask = cv2.inRange(hsv_mask, lower_color, upper_color)

                # increasing standard deviation to blur more (repairing the mask)
                blurred_mask = cv2.GaussianBlur(
                    color_mask, (7, 7), sigmaX=1, sigmaY=1)

                # applying dilation (optional) and erosion to the mask
                kernel = np.ones((3, 3), np.uint8)
                # dilated_mask = cv2.dilate(mask, dilation_kernel, iterations=1)
                eroded_mask = cv2.erode(blurred_mask, kernel, iterations=1)

                components[label] = eroded_mask

                # plt.imshow(eroded_mask, cmap='gray')
                # plt.show()
    else:  # binary mask
        components = {}

        # finding the components in the binary image
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            image, connectivity=8)

        for label in range(1, num_labels):
            # creating a mask for the selected component
            component_mask = np.zeros(image.shape, dtype=np.uint8)
            component_mask[labels == label] = 255

            binary_mask = component_mask[:, :, 0]
            # increasing standard deviation to blur more (repairing the mask)
            blurred_mask = cv2.GaussianBlur(
                binary_mask, (7, 7), sigmaX=1, sigmaY=1)

            # applying dilation (optional) and erosion to the mask
            kernel = np.ones((3, 3), np.uint8)
            # dilated_mask = cv2.dilate(mask, dilation_kernel, iterations=1)
            eroded_mask = cv2.erode(blurred_mask, kernel, iterations=1)

            components[label] = eroded_mask

    print('\033[94m', "\n Number of objects detected: ",
          len(components), '\033[0m')
    # Returning the dictionary of masks
    return components


def multiple_object_annotation_color(annotation_color, threshold=0.3):
    # extracting color channels
    red, green, blue = annotation_color

    # Using passed color:
    # generating a random color
    # random_red = random.uniform(-threshold * 255, threshold * 255)
    # random_green = random.uniform(-threshold * 255, threshold * 255)
    # random_blue = random.uniform(-threshold * 255, threshold * 255)

    # adding the random color to the annotation color
    # new_red = max(0, min(255, red + random_red))
    # new_green = max(0, min(255, green + random_green))
    # new_blue = max(0, min(255, blue + random_blue))
    # return (new_red, new_green, new_blue)

    # Using random bright colors:
    # generating a random color
    hue = random.uniform(0, 360)

    # converting the HSV color to RGB
    hsv_color = (hue / 360, 1, 1)
    random_rgb = tuple(int(i * 255) for i in colorsys.hsv_to_rgb(*hsv_color))

    # returning the new color
    return random_rgb
