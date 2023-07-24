import cv2
import numpy as np
import matplotlib.pyplot as plt


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
        if area > 40:  # removing small noise
            # Approximating the polygon to reduce the number of points
            approx_contour = cv2.approxPolyDP(
                contour, epsilon * cv2.arcLength(contour, True), True)
            sorted_contours.append(approx_contour)

    print("Number of contours: ", len(sorted_contours))

    # sorting the contours based on the y coordinate of the bounding box
    sorted_contours = sorted(
        sorted_contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    return sorted_contours


<<<<<<< HEAD
def multiple_objects_polygon_approximation(mask, epsilon, do_cvt):
    # Image Cleaning
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

    # splitting the different objects in the mask
    num_labels, labels_im, _, _ = cv2.connectedComponentsWithStats(
        eroded_mask, connectivity=8)

    # Polygon approximation
    object_contours = {}

    # iterating over all the labels from 1 to num_labels (inclusive)
    for target_label in range(1, num_labels + 1):
        # creating a binary mask containing only the pixels belonging to the selected connected component
        target_mask = np.uint8(labels_im == target_label) * 255

        # retrieving the contours of the selected connected component
        contours, _ = cv2.findContours(
            target_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # looping through the contours
        for contour in contours:
            # calculating the area of the contour
            area = cv2.contourArea(contour)
            if area > 40:  # removing small noise
                # approximating the polygon to reduce the number of points
                approx_contour = cv2.approxPolyDP(
                    contour, epsilon * cv2.arcLength(contour, True), True)

                # adding the contour to the dictionary
                if target_label not in object_contours:
                    object_contours[target_label] = []
                object_contours[target_label].append(approx_contour)

    return object_contours


def single_object_k_means_clustering(mask, epsilon, max_clusters, do_cvt):
    # transforming image into a binary image
    if do_cvt:
        # transforming image into a binary image
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # thresholding the image
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
=======
def k_means_clustering(mask, epsilon, max_clusters, do_cvt):
    # transforming image into a binary image
    if do_cvt:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
>>>>>>> 08ae8ecf571e5c59de6a536c47c3d059dee2f2fc

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


def multiple_objects_k_means_clustering(mask, epsilon, max_clusters, do_cvt):
    # Image Cleaning
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

    # splitting the different objects in the mask
    num_labels, labels_im, _, _ = cv2.connectedComponentsWithStats(
        eroded_mask, connectivity=8)

    # K-means clustering
    # initializing a dictionary to store the contours for each connected component
    annotations = {}

    # iterating over all the labels from 1 to num_labels (inclusive)
    for target_label in range(1, num_labels + 1):
        # creating a binary mask containing only the pixels belonging to the selected connected component
        target_mask = np.uint8(labels_im == target_label) * 255

        # retrieving the contours of the selected connected component
        contours, _ = cv2.findContours(
            target_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

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
        annotations[target_label] = [convex_hull]

    return annotations
