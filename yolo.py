import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import os


def mask_to_annotation(mask):
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


def display(im_dict, annotation_color):
    # Displaying bounding boxes on the image
    image_with_bounding_box = im_dict['image'].copy()
    for contour in im_dict['contours']:
        x, y, w, h = contour
        cv2.rectangle(image_with_bounding_box, (x, y),
                      (x+w, y+h), annotation_color, 7)

    # Displaying original mask on the left and annotation on the right
    plt.rcParams["figure.figsize"] = (20, 10)

    plt.subplot(121)
    plt.rcParams['axes.titlesize'] = 20
    plt.title('Original mask')
    plt.imshow(im_dict['image'], interpolation='nearest')
    plt.axis('off')

    plt.subplot(122)
    plt.rcParams['axes.titlesize'] = 20
    plt.title('Annotation')
    plt.imshow(image_with_bounding_box, interpolation='nearest')
    plt.axis('off')
    plt.show()


def save(im_dict):
    # Creating directories if they don't exist
    if not os.path.exists(im_dict['directory']):
        os.makedirs(im_dict['directory'])

    if not os.path.exists(im_dict['directory']+"/"+im_dict['file_name']):
        os.makedirs(im_dict['directory']+"/"+im_dict['file_name'])

    # Saving the annotation in YOLO text file format
    file_path = os.path.join(
        "./"+im_dict['directory']+"/"+im_dict['file_name'], str(im_dict['file_name']) + '.txt')

    with open(file_path, 'w') as f:
        for count, contour in enumerate(im_dict['contours']):
            # Formatting to account for YOLO format
            x, y, w, h = contour
            x = x/im_dict['image'].shape[1]
            y = y/im_dict['image'].shape[0]
            w = w/im_dict['image'].shape[1]
            h = h/im_dict['image'].shape[0]

            x = x + w / 2
            y = y + h / 2

            f.write("0 " + str(x) + " " +
                    str(y) + " " + str(w) + " " + str(h)+"\n")
        f.close()

    # Saving the category in a text file
    labels_file_path = os.path.join(
        "./"+im_dict['directory']+"/"+im_dict['file_name'], 'labels.txt')
    with open(labels_file_path, 'w') as f:
        f.write(im_dict['category'])
        f.close()


def annotate(im, do_display=True, do_save=True, annotation_color='g'):
    # Retrieving parameters from the tuple
    id_, name, image, project_name, category, directory = im

    print("\n Annotating image: ", name)

    # Creating a dictionary to store the image and its annotations
    im_dict = {}
    im_dict['file_name'] = os.path.splitext(name)[0]
    im_dict['image'] = image
    im_dict['category'] = category
    im_dict['contours'] = mask_to_annotation(image)
    im_dict['directory'] = directory

    # Displaying and saving the image, depending on the passed parameters
    if do_display:
        display(im_dict, annotation_color)

    if do_save:
        save(im_dict)
        print('\033[92m', "Succesfully saved image: ", name, '\033[0m\n\n')
    print("-"*120)

    return im_dict
