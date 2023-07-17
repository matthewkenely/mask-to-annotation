import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import os


project_name = "COTS Dataset"
category = "directory"


def contours_to_coco(contours):
    coco_data = {
          "info": {
            "description": project_name
          },
          "images": [],
        "annotations": [],
        "categories": []
    }
    
    annotation_id = 1
    
    for image_id,image_descriptor in contours.items():
        contour_list = image_descriptor["contours"]
        
        # Create image entry
        image_entry = {
            "id": image_id,
            "width": image_descriptor["width"],
            "height": image_descriptor["height"],
            "file_name": image_descriptor["file_name"]
        }
        
        # Create annotation entry
        for contour in contour_list:
            contour = np.array(contour, dtype=np.float32)

            # Check if the contour has enough points
            if contour.shape[0] < 3:
                continue

            # Create annotation entry
            annotation_entry = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": image_id,
                "segmentation": contour.squeeze().tolist(),
                "area": cv2.contourArea(contour),
                "bbox": cv2.boundingRect(contour)
            }
            

        
        # Create categories entry
#         category_entry = {
#             "id": annotation_id,
#             "name": bbox
#         }
        
        coco_data["images"].append(image_entry)
        coco_data["annotations"].append(annotation_entry)
        
        annotation_id += 1
    
    return coco_data
