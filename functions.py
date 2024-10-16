import os
import random
import numpy as np
from PIL import Image

# Retrieve random image and mask pair (default = 1), returns in numpy
def get_image_and_mask(num_samples=1):
    # Get list of all image and mask files
    image_files = [f for f in os.listdir(os.path.join("dataset", "images")) if f.endswith('.jpg')]
    mask_files = [f for f in os.listdir(os.path.join("dataset", "masks")) if f.endswith('.jpg')]
    
    # Ensure both lists are sorted to match corresponding images/masks
    image_files.sort()
    mask_files.sort()
    
    # Randomly select num_samples indices
    selected_indices = random.sample(range(len(image_files)), num_samples)
    
    # Retrieve the selected images and masks
    for idx in selected_indices:
        image_path = os.path.join(os.path.join("dataset", "images"), image_files[idx])
        mask_path = os.path.join(os.path.join("dataset", "masks"), mask_files[idx])
        
        # Load the image and mask as numpy
        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path))
    
    return image, mask

def load_dataset():
    image_files = sorted([f for f in os.listdir(os.path.join("dataset", "images")) if f.endswith('.jpg')])
    mask_files = sorted([f for f in os.listdir(os.path.join("dataset", "masks")) if f.endswith('.jpg')])

    images = [np.array(Image.open(os.path.join("dataset", "images", f))) for f in image_files]
    masks = [np.array(Image.open(os.path.join("dataset", "masks", f))) for f in mask_files]

    return images, masks