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
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Looping through the contours and finding the bounding boxes
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        current_box = (x, y, w, h)
        is_nested = False

        # Checking if the current bounding box is nested inside another bounding box
        # If yes, then current bounding box is ignored
        for box in bounding_boxes:
            if box[0] <= x and box[1] <= y and (box[0] + box[2]) >= (x + w) and (box[1] + box[3]) >= (y + h):
                is_nested = True
                break

        if not is_nested:
            bounding_boxes.append(current_box)

    # Sorting bounding boxes based on area in descending order
    bounding_boxes = sorted(
        bounding_boxes, key=lambda box: box[2] * box[3], reverse=True)

    # Taking the first two non-nested bounding boxes
    bounding_boxes = bounding_boxes[:2]

    return bounding_boxes


def display(im_dict, annotation_color):
    # Displaying bounding boxes on the image
    image_with_bounding_box = im_dict['image'].copy()
    for contour in im_dict['contours']:
        x, y, w, h = contour
        cv2.rectangle(image_with_bounding_box, (x, y),
                      (x+w, y+h), annotation_color, 8)

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
