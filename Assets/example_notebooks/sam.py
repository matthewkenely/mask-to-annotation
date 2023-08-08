# Dependencies required for segment-anything
# %pip install git+https://github.com/facebookresearch/segment-anything.git
# %pip install torch torchvision
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
torch.cuda.empty_cache()
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Segment-anything model
# Downloading model checkpoint:
# Utilise the following link: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# Place the downloaded checkpoint in a new directory named Sam_checkpoints
sam_checkpoint = "Sam_checkpoints/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_batch=16
)


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


image = cv2.imread('COTSDataset/Part 2 - Multiple Objects/academic_book_no/masks/ac_3_colour_mask_7_mask.png')
# reduce image size to 256 x 256
# image = cv2.resize(image, (256, 256))
# plt.imshow(image)

from time import time

start = time()

masks = mask_generator.generate(image)

end = time()
print(f'SAM time: {end - start}')

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 

print(f'SAM time: {end - start}')