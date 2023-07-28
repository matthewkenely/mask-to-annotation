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
        # setting the transparency of the filled bounding box
        alpha = 0.25
        # sorting contours by area
        for label, contours_list in im_dict['contours'].items():
            im_dict['contours'][label] = sorted(
                contours_list, key=lambda x: cv2.contourArea(x))
        # drawing each contour on the blank image with the specified annotation_color
        for label, contours in im_dict['contours'].items():
            # creating a blank image
            blank_image = np.zeros_like(im_dict['image'])
            # getting the annotation color
            annotation_color = ah.multiple_object_annotation_color(
                annotation_color=annotation_color)
            for contour in contours:
                # drawing the contour on the blank image
                contour_image = cv2.drawContours(blank_image, [contour], -1,
                                                 annotation_color, 7, cv2.LINE_AA)
                # adding the contour image to the annotated image
                contours_image = cv2.drawContours(annotated_image.copy(), [contour], -1,
                                                  annotation_color, 7, cv2.LINE_AA)
                # adding filled contour to the annotated image
                filled_contour_image = cv2.drawContours(contours_image.copy(), [contour], -1,
                                                        annotation_color, cv2.FILLED, cv2.LINE_AA)
                # adding the contour image to the annotated image
                annotated_image = cv2.addWeighted(annotated_image,
                                                  1-alpha, filled_contour_image, alpha, 0)

                # adding the contours to the annotated image
                annotated_image = cv2.drawContours(annotated_image, [contour], -1,
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
    # creating a dictionary in COCO format
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
        'categories': []
    }

    if (object_configuration == SINGLE_OBJ):
        # adding the category to the dictionary
        coco_data['categories'].append(
            {
                'id': im_dict['id'],
                'name': im_dict['category']
            })

        # looping through the contours and adding them to the dictionary
        for contour in im_dict['contours']:
            contour = np.array(contour, dtype=np.float32)

            # checking if the contour has enough points
            if contour.shape[0] < 3:
                continue

            # adding the contour to the dictionary
            coco_data['annotations'].append({
                'id': im_dict['id'],
                'iscrowd': 0,
                'image_id': im_dict['id'],
                'category_id': im_dict['id'],
                'segmentation': [contour.flatten().tolist()],
                'bbox': cv2.boundingRect(contour),
                'area': cv2.contourArea(contour)
            })
    else:
        counter = 0
        # Looping through the contours and adding them to the dictionary
        for label, contours_list in im_dict['contours'].items():
            # retrieving the category id and label
            category_id = counter
            category_label = im_dict['category']+str(category_id)

            # looping through the contours
            for contour in contours_list:
                contour = np.array(contour, dtype=np.float32)

                # Checking if the contour has enough points
                if contour.shape[0] < 3:
                    continue

                # Adding the contour to the dictionary
                coco_data['annotations'].append({
                    'id': counter,
                    'iscrowd': 0,
                    'image_id': im_dict['id'],
                    'category_id': category_id,
                    'segmentation': [contour.flatten().tolist()],
                    'bbox': cv2.boundingRect(contour),
                    'area': cv2.contourArea(contour)
                })

            # adding the category to the dictionary
            coco_data['categories'].append(
                {
                    'id': category_id,
                    'name': category_label
                })
            counter += 1

    # creating a directory to store the annotations
    if not os.path.exists(im_dict['directory']):
        os.makedirs(im_dict['directory'])

    # saving the annotations in COCO JSON file format
    file_path = os.path.join(
        "./"+im_dict['directory'], str(os.path.splitext(im_dict['file_name'])[0]) + '.json')

    with open(file_path, 'w') as f:
        json.dump(coco_data, f, indent=4)


def annotate(im, do_display=True, do_save=True, do_print=True, annotation_color=(255, 0, 255), epsilon=0.005, configuration=POLY_APPROX, object_configuration=SINGLE_OBJ, do_cvt=True):
    # retrieving parameters from the tuple
    id_, name, image, project_name, category, directory = im

    if do_print:
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
        if do_print:
            print('\033[92m', "Succesfully saved image: ", name, '\033[0m\n\n')

    if do_print:
        print("-"*120)

    return im_dict
