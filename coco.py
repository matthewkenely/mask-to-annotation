import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import os


def mask_to_annotation(mask, epsilon):
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
        approx_contour = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
        sorted_contours.append(approx_contour)

    # Sorting the contours based on the y coordinate of the bounding box
    sorted_contours = sorted(
        sorted_contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    return sorted_contours


def display(im_dict, annotation_color):
    # Displaying the contours on the image
    annotated_image = im_dict['image'].copy()
    cv2.drawContours(annotated_image, im_dict['contours'], -1,
                     annotation_color, 7, cv2.LINE_AA)

    # Display original mask on the left and annotation on the right
    plt.rcParams["figure.figsize"] = (20, 10)

    plt.subplot(121)
    plt.rcParams['axes.titlesize'] = 20
    plt.title('Original mask')
    plt.imshow(im_dict['image'], interpolation='nearest')
    plt.axis('off')

    plt.subplot(122)
    plt.rcParams['axes.titlesize'] = 20
    plt.title('Annotation')
    plt.imshow(annotated_image, interpolation='nearest')
    plt.axis('off')
    plt.show()


def save(im_dict):
    # Creating a dictionary in COCO format
    coco_data = {
        'info': {
            'description': im_dict['project_name']
        },
        'images': [
            {
                'id': im_dict['id'],
                'width': im_dict['width'],
                'height': im_dict['height'],
                'file_name': im_dict['file_name']
            }
        ],
        'annotations': [],
        'categories': [
            {
                'id': im_dict['id'],
                'name': im_dict['category']
            }
        ]
    }

    # Looping through the contours and adding them to the dictionary
    for contour in im_dict['contours']:
        contour = np.array(contour, dtype=np.float32)

        # Checking if the contour has enough points
        if contour.shape[0] < 3:
            continue

        # Adding the contour to the dictionary
        coco_data['annotations'].append({
            'id': im_dict['id'],
            'iscrowd': 0,
            'image_id': im_dict['id'],
            'category_id': im_dict['id'],
            'segmentation': [contour.flatten().tolist()],
            'bbox': cv2.boundingRect(contour),
            'area': cv2.contourArea(contour)
        })

        # Creating a directory to store the annotations
        if not os.path.exists(im_dict['directory']):
            os.makedirs(im_dict['directory'])

        # Saving the annotations in COCO JSON file format
        file_path = os.path.join(
            "./"+im_dict['directory'], str(os.path.splitext(im_dict['file_name'])[0]) + '.json')

        with open(file_path, 'w') as f:
            json.dump(coco_data, f, indent=4)


def annotate(im, do_display=True, do_save=True, do_print=True, annotation_color='g', epsilon=0.024):
    # Retrieving parameters from the tuple
    id_, name, image, project_name, category, directory = im

    if do_print:
        print("\n Annotating image: ", name)

    # Creating a dictionary to store the image and its annotations
    im_dict = {}
    im_dict['id'] = 0  # id_
    im_dict['file_name'] = name
    im_dict['image'] = image
    im_dict['width'] = image.shape[1]
    im_dict['height'] = image.shape[0]
    im_dict['contours'] = mask_to_annotation(image, epsilon)
    im_dict['project_name'] = project_name
    im_dict['category'] = category
    im_dict['directory'] = directory

    # Displaying and saving the image, depending on the passed parameters
    if do_display:
        display(im_dict, annotation_color)

    if do_save:
        save(im_dict)
        if do_print:
            print('\033[92m', "Succesfully saved image: ", name, '\033[0m\n\n')

    if do_print:
        print("-"*120)

    return im_dict
