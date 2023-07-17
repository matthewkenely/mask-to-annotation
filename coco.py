import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import os


project_name = 'COTS Dataset'
category = 'directory'


# Load images
images = {}
for x in os.listdir('masks'):
    images[x] = cv2.imread('masks/' + x)
    images[x] = images[x][:, :, ::-1]
    plt.imshow(images[x], interpolation='nearest')
    plt.axis('off')
    plt.show()


def mask_to_annotation(mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(
        contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
    return sorted_contours


def contours_to_coco(contours):
    coco_data = {
        'info': {
            'description': project_name
        },
        'images': [],
        'annotations': [],
        'categories': []
    }

    annotation_id = 1

    for image_id, image_descriptor in contours.items():
        contour_list = image_descriptor['contours']

        # Create image entry
        image_entry = {
            'id': image_id,
            'width': image_descriptor['width'],
            'height': image_descriptor['height'],
            'file_name': image_descriptor['file_name']
        }

        # Create annotation entry
        for contour in contour_list:
            contour = np.array(contour, dtype=np.float32)

            # Check if the contour has enough points
            if contour.shape[0] < 3:
                continue

            # Create annotation entry
            annotation_entry = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': image_id,
                'segmentation': contour.squeeze().tolist(),
                'area': cv2.contourArea(contour),
                'bbox': cv2.boundingRect(contour)
            }

        # Create categories entry
#         category_entry = {
#             'id': annotation_id,
#             'name': bbox
#         }

        coco_data['images'].append(image_entry)
        coco_data['annotations'].append(annotation_entry)

        annotation_id += 1

    return coco_data


def display():
    pass


def save(contours, id_):
    coco_data = {
        'info': {
            'description': project_name
        },
        'images': [],
        'annotations': [],
        'categories': []
    }

    annotation_id = 1

    for image_id, image_descriptor in contours.items():
        contour_list = image_descriptor['contours']

        # Create image entry
        image_entry = {
            'id': image_id,
            'width': image_descriptor['width'],
            'height': image_descriptor['height'],
            'file_name': image_descriptor['file_name']
        }

        # Create annotation entry
        for contour in contour_list:
            contour = np.array(contour, dtype=np.float32)

            # Check if the contour has enough points
            if contour.shape[0] < 3:
                continue

            # Create annotation entry
            annotation_entry = {
                'image_id': image_id,
                'category_id': image_id,
                'segmentation': contour.squeeze().tolist(),
                'area': cv2.contourArea(contour),
                'bbox': cv2.boundingRect(contour)
            }

        # Create categories entry
#         category_entry = {
#             'id': annotation_id,
#             'name': bbox
#         }

        coco_data['images'].append(image_entry)
        coco_data['annotations'].append(annotation_entry)

        annotation_id += 1

    with open(id_ + '.json', 'w') as f:
        json.dump(coco_data, f, indent=4)


def annotate(im, display=True, save=True):
    id_, name, image = im

    im_dict = {}
    im_dict['id'] = id_
    im_dict['file_name'] = name
    im_dict['width'] = image.shape[1]
    im_dict['height'] = image.shape[0]
    im_dict['contours'] = mask_to_annotation(image)

    if display:
        display()

    if save:
        save(im_dict, id_)
