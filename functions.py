import os
import random
import numpy as np
from PIL import Image

# Retrieve random image and mask pair (default = 1) from list, returns in numpy
def get_image_and_mask(images, masks, num_samples=1):
    # Ensure the lists are of the same length
    if len(images) != len(masks):
        raise ValueError("The number of images and masks must be equal.")

    # Check if num_samples is valid
    if num_samples > len(images):
        raise ValueError("num_samples cannot exceed the number of available images/masks.")

    # Randomly select 'num_samples' indices
    selected_indices = random.sample(range(len(images)), num_samples)

    # Retrieve the selected images and masks
    selected_images = [images[i] for i in selected_indices]
    selected_masks = [masks[i] for i in selected_indices]

    # Return a single pair if num_samples = 1, else return lists
    if num_samples == 1:
        return selected_images[0], selected_masks[0]
    else:
        return selected_images, selected_masks

def load_dataset():
    image_files = sorted([f for f in os.listdir(os.path.join("dataset", "images")) if f.endswith('.jpg')])
    mask_files = sorted([f for f in os.listdir(os.path.join("dataset", "masks")) if f.endswith('.jpg')])

    images = [np.array(Image.open(os.path.join("dataset", "images", f))) for f in image_files]
    masks = [np.array(Image.open(os.path.join("dataset", "masks", f))) for f in mask_files]

    return images, masks