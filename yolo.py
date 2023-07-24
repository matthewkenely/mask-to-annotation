import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import annotation_helper as ah

# Constants
# Single objects
SINGLE_OBJ = 0
# Multiple objects
MULTIPLE_OBJ = 1


def mask_to_annotation(mask, object_configuration, do_cvt):
    # checking the configuration
    if object_configuration == SINGLE_OBJ:
        return ah.single_object_bounding_box(mask, do_cvt)
    elif object_configuration == MULTIPLE_OBJ:
        return ah.multiple_objects_bounding_box(mask, do_cvt)
    else:
        pass


def display(im_dict, annotation_color):
    # displaying bounding boxes on the image
    image_with_bounding_box = im_dict['image'].copy()
    for contour in im_dict['contours']:
        x, y, w, h = contour
        cv2.rectangle(image_with_bounding_box, (x, y),
                      (x+w, y+h), annotation_color, 7)

    # displaying original mask on the left and annotation on the right
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
    # creating directories if they don't exist
    if not os.path.exists(im_dict['directory']):
        os.makedirs(im_dict['directory'])

    if not os.path.exists(im_dict['directory']+"/"+im_dict['file_name']):
        os.makedirs(im_dict['directory']+"/"+im_dict['file_name'])

    # saving the annotation in YOLO text file format
    file_path = os.path.join(
        "./"+im_dict['directory']+"/"+im_dict['file_name'], str(im_dict['file_name']) + '.txt')

    with open(file_path, 'w') as f:
        for count, contour in enumerate(im_dict['contours']):
            # formatting to account for YOLO format
            x, y, w, h = contour
            x = x/im_dict['image'].shape[1]
            y = y/im_dict['image'].shape[0]
            w = w/im_dict['image'].shape[1]
            h = h/im_dict['image'].shape[0]

            x = x + w / 2
            y = y + h / 2

            f.write(str(count)+" " + str(x) + " " +
                    str(y) + " " + str(w) + " " + str(h)+"\n")
        f.close()

    # saving the category in a text file
    labels_file_path = os.path.join(
        "./"+im_dict['directory']+"/"+im_dict['file_name'], 'labels.txt')
    with open(labels_file_path, 'w') as f:
        f.write(im_dict['category'])
        f.close()


def annotate(im, do_display=True, do_save=True, annotation_color=(0, 255, 0), object_configuration=SINGLE_OBJ, do_cvt=True):
    # retrieving parameters from the tuple
    id_, name, image, project_name, category, directory = im

    print("\n Annotating image: ", name)

    # creating a dictionary to store the image and its annotations
    im_dict = {}
    im_dict['file_name'] = os.path.splitext(name)[0]
    im_dict['image'] = image
    im_dict['category'] = category
    im_dict['contours'] = mask_to_annotation(
        image, object_configuration, do_cvt)
    im_dict['directory'] = directory

    # displaying and saving the image, depending on the passed parameters
    if do_display:
        display(im_dict, annotation_color)

    if do_save:
        save(im_dict)
        print('\033[92m', "Succesfully saved image: ", name, '\033[0m\n\n')
    print("-"*120)

    return im_dict
