import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import os


def mask_to_annotation(mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # increase standard deviation to blur more
    mask = cv2.GaussianBlur(mask, (7, 7), sigmaX=1, sigmaY=1)

    dilation_kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(mask, dilation_kernel, iterations=3)
    eroded_mask = cv2.erode(dilated_mask, dilation_kernel, iterations=1)
    contours, _ = cv2.findContours(
        eroded_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    sorted_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2000:  # Example area threshold
            # Approximate the contour to reduce the number of points
            # Adjust the epsilon value as needed
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            sorted_contours.append(approx_contour)

    sorted_contours = sorted(
        sorted_contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    return sorted_contours


def display(im_dict, annotation_color):
    for i in range(len(im_dict['contours'])):
        x = im_dict['contours'][i][:, 0, 0]
        y = im_dict['contours'][i][:, 0, 1]

    # Display original mask on the left and annotation on the right
    # increase size
    plt.rcParams["figure.figsize"] = (20, 10)

    plt.subplot(121)
    plt.rcParams['axes.titlesize'] = 20
    plt.title('Original mask')
    plt.imshow(im_dict['image'], interpolation='nearest')
    plt.axis('off')

    plt.subplot(122)
    plt.rcParams['axes.titlesize'] = 20
    plt.title('Annotation')
    plt.imshow(im_dict['image'], interpolation='nearest')
    plt.plot(x, y, annotation_color, linewidth=3)
    plt.axis('off')
    plt.show()


def save(im_dict):
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

    for contour in im_dict['contours']:
        contour = np.array(contour, dtype=np.float32)

        # Check if the contour has enough points
        if contour.shape[0] < 3:
            continue

        vgg_data[str(im_dict['file_name'])]["regions"]["0"]["shape_attributes"]["all_points_x"] = [
            int(contour[i][0][0]) for i in range(contour.shape[0])]
        vgg_data[str(im_dict['file_name'])]["regions"]["0"]["shape_attributes"]["all_points_y"] = [
            int(contour[i][0][1]) for i in range(contour.shape[0])]

        if not os.path.exists(im_dict['directory']):
            os.makedirs(im_dict['directory'])

        file_path = os.path.join(
            "./"+im_dict['directory'], str(os.path.splitext(im_dict['file_name'])[0]) + '.json')

        with open(file_path, 'w') as f:
            json.dump(vgg_data, f, indent=4)


def annotate(im, do_display=True, do_save=True, annotation_color='g'):
    id_, name, image, project_name, category, directory = im

    print("\n Annotating image: ", name)

    im_dict = {}
    im_dict['id'] = 0  # id_
    im_dict['file_name'] = name
    im_dict['image'] = image
    im_dict['width'] = image.shape[1]
    im_dict['height'] = image.shape[0]
    im_dict['contours'] = mask_to_annotation(image)
    im_dict['project_name'] = project_name
    im_dict['category'] = category
    im_dict['directory'] = directory

    if do_display:
        display(im_dict, annotation_color)

    if do_save:
        save(im_dict)
        print('\033[92m', "Succesfully saved image: ", name, '\033[0m\n\n')
    print("-"*120)
