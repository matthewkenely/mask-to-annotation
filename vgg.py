import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import annotation_helper as ah

# Constants
# Polygon approximation
PA = 0
# K-means clustering
KMC = 1
# Segment-anything
SA = 2


def mask_to_annotation(mask, epsilon, configuration):
    # checking the configuration
    if configuration == PA:
        return ah.polygon_approximation(mask, epsilon)
    elif configuration == KMC:
        return ah.k_means_clustering(mask, epsilon, max_clusters=10)
    elif configuration == SA:
        return ah.segment_anything(mask)
    else:
        pass


def display(im_dict, annotation_color):
    # displaying the contours on the image
    annotated_image = im_dict['image'].copy()
    cv2.drawContours(annotated_image, im_dict['contours'], -1,
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


def save(im_dict):
    # creating a dictionary in the VGG format
    vgg_data = {
        str(im_dict['file_name']): {
            "fileref": "",
            "size": im_dict['width'],
            "filename": im_dict['file_name'],
            "base64_img_data": "",
            "file_attributes": {
            },
            "regions": {
                "0": {
                    "shape_attributes": {
                        "name": "polygon",
                        "all_points_x": [],
                        "all_points_y": []
                    },
                    "region_attributes": {
                        "label": im_dict['category']
                    }
                }
            }

        }
    }

    # looping through the contours and adding the points to the dictionary
    for contour in im_dict['contours']:
        contour = np.array(contour, dtype=np.float32)

        # checking if the contour has enough points (error handling)
        if contour.shape[0] < 3:
            continue

        vgg_data[str(im_dict['file_name'])]["regions"]["0"]["shape_attributes"]["all_points_x"] = [
            int(contour[i][0][0]) for i in range(contour.shape[0])]
        vgg_data[str(im_dict['file_name'])]["regions"]["0"]["shape_attributes"]["all_points_y"] = [
            int(contour[i][0][1]) for i in range(contour.shape[0])]

        # creating a directory to store the annotations
        if not os.path.exists(im_dict['directory']):
            os.makedirs(im_dict['directory'])

        # saving the annotations in VGG JSON file format
        file_path = os.path.join(
            "./"+im_dict['directory'], str(os.path.splitext(im_dict['file_name'])[0]) + '.json')

        with open(file_path, 'w') as f:
            json.dump(vgg_data, f, indent=4)


def annotate(im, do_display=True, do_save=True, annotation_color=(255, 0, 0), epsilon=0.005, configuration=PA):
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
    im_dict['contours'] = mask_to_annotation(image, epsilon)
    im_dict['project_name'] = project_name
    im_dict['category'] = category
    im_dict['directory'] = directory

    # displaying and saving the image, depending on the passed parameters
    if do_display:
        display(im_dict, annotation_color)

    if do_save:
        save(im_dict)
        print('\033[92m', "Succesfully saved image: ", name, '\033[0m\n\n')
    print("-"*120)

    return im_dict
