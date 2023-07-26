import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import annotation_helper as ah

# Constants
# Polygon approximation
POLY_APPROX = 0
# K-means clustering
K_MEANS_CLUSTER = 1
# Single objects
SINGLE_OBJ = 0
# Multiple objects
MULTIPLE_OBJ = 1


def mask_to_annotation(mask, epsilon, configuration, object_configuration, do_cvt):
    # checking the configuration
    if configuration == POLY_APPROX and object_configuration == SINGLE_OBJ:
        return ah.single_object_polygon_approximation(mask, epsilon, do_cvt)
    elif configuration == POLY_APPROX and object_configuration == MULTIPLE_OBJ:
        return ah.multiple_objects_polygon_approximation(mask, epsilon, do_cvt)
    elif configuration == K_MEANS_CLUSTER and object_configuration == SINGLE_OBJ:
        return ah.single_object_k_means_clustering(mask, max_clusters=100, do_cvt=do_cvt)
    elif configuration == K_MEANS_CLUSTER and object_configuration == MULTIPLE_OBJ:
        return ah.multiple_objects_k_means_clustering(mask, max_clusters=100, do_cvt=do_cvt)
    else:
        pass


def display(im_dict, annotation_color, object_configuration):
    # displaying the contours on the image
    annotated_image = im_dict['image'].copy()
    if (object_configuration == SINGLE_OBJ):
        cv2.drawContours(annotated_image, im_dict['contours'], -1,
                         annotation_color, 7, cv2.LINE_AA)
    else:
        # drawing each contour on the blank image with the specified annotation_color
        for label, contours in im_dict['contours'].items():
            for contour in contours:
                cv2.drawContours(annotated_image, [contour], -1,
                                 annotation_color, 7, cv2.LINE_AA)

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
    plt.imshow(annotated_image, interpolation='nearest')
    plt.axis('off')
    plt.show()


def save(im_dict, object_configuration):
    # creating a dictionary in the VGG format
    vgg_data = {
        str(im_dict['file_name']): {
            "fileref": "",
            "size": im_dict['width'],
            "filename": im_dict['file_name'],
            "base64_img_data": "",
            "file_attributes": {},
            "regions": {}
        }
    }

    if (object_configuration == SINGLE_OBJ):
        # flattening the nested structure and convert contours to a list of (x, y) coordinate tuples
        contour_points = [tuple(coord[0])
                          for contour in im_dict['contours'] for coord in contour]

        # creating a dictionary for the region data of each contour
        region_data = {
            "shape_attributes": {
                "name": "polygon",
                "all_points_x": [int(coord[0]) for coord in contour_points],
                "all_points_y": [int(coord[1]) for coord in contour_points]
            },
            "region_attributes": {
                "label": "1"
            }
        }
        # adding the region data to the dictionary
        vgg_data[str(im_dict['file_name'])]["regions"]["0"] = region_data
    else:
        # looping through the contours and adding them to the dictionary
        for idx, contours in im_dict['contours'].items():
            # flattening the nested structure and convert contours to a list of (x, y) coordinate tuples
            contour_points = [tuple(coord[0])
                              for contour in contours for coord in contour]

            # checking if the contour has enough points (error handling)
            if len(contour_points) < 3:
                continue

            # creating a dictionary for the region data of each contour
            region_data = {
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": [int(coord[0]) for coord in contour_points],
                    "all_points_y": [int(coord[1]) for coord in contour_points]
                },
                "region_attributes": {
                    # taking the label from the index of the contour
                    "label": idx
                }
            }
            # adding the region data to the dictionary
            vgg_data[str(im_dict['file_name'])
                     ]["regions"][str(idx-1)] = region_data

    # creating a directory to store the annotations
    if not os.path.exists(im_dict['directory']):
        os.makedirs(im_dict['directory'])

    # saving the annotations in VGG JSON file format
    file_path = os.path.join(
        im_dict['directory'], str(os.path.splitext(im_dict['file_name'])[0]) + '.json')

    with open(file_path, 'w') as f:
        json.dump(vgg_data, f, indent=4)


def annotate(im, do_display=True, do_save=True, annotation_color=(255, 0, 0), epsilon=0.005, configuration=POLY_APPROX, object_configuration=SINGLE_OBJ, do_cvt=True):
    # retrieving parameters from the tuple
    id_, name, image, project_name, category, directory = im

    print("\n Annotating image: ", name)

    # creating a dictionary to store the image and its annotations
    im_dict = {}
    im_dict['id'] = 0  # id_
    im_dict['file_name'] = name
    im_dict['image'] = image
    im_dict['width'] = image.shape[1]
    im_dict['height'] = image.shape[0]
    im_dict['contours'] = mask_to_annotation(
        image, epsilon, configuration, object_configuration, do_cvt)
    im_dict['project_name'] = project_name
    im_dict['category'] = category
    im_dict['directory'] = directory

    # displaying and saving the image, depending on the passed parameters
    if do_display:
        display(im_dict, annotation_color, object_configuration)

    if do_save:
        save(im_dict, object_configuration)
        print('\033[92m', "Succesfully saved image: ", name, '\033[0m\n\n')
    print("-"*120)

    return im_dict
