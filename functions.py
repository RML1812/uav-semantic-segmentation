import os
import random
import numpy as np
import cv2
from PIL import Image
from skimage.color import rgb2lab
from sklearn.metrics import v_measure_score
from skimage.feature import local_binary_pattern
from skimage.filters import sobel

global dataset_path
dataset_path = "./dataset"

# Retrieve random image and mask pair (default = 1), returns in numpy
def get_image_and_mask(num_samples=1):
    # Get list of all image and mask files
    image_files = [f for f in os.listdir(os.path.join(dataset_path, "images")) if f.endswith('.jpg')]
    mask_files = [f for f in os.listdir(os.path.join(dataset_path, "masks")) if f.endswith('.jpg')]
    
    # Ensure both lists are sorted to match corresponding images/masks
    image_files.sort()
    mask_files.sort()
    
    # Randomly select num_samples indices
    selected_indices = random.sample(range(len(image_files)), num_samples)
    
    # Retrieve the selected images and masks
    for idx in selected_indices:
        image_path = os.path.join(os.path.join(dataset_path, "images"), image_files[idx])
        mask_path = os.path.join(os.path.join(dataset_path, "masks"), mask_files[idx])
        
        # Load the image and mask as numpy
        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path))
    
    return image, mask

# Extracting features from image, returns in column stack
def extract_features(image: np.ndarray):
    image = cv2.fastNlMeansDenoisingColored(image, None, 21, 21, 10, 21)

    # Convert RGB to Lab color space
    lab_image = rgb2lab(image)  # Lab has 3 channels: L, a, b

    # Extract Lab channels
    l_channel = lab_image[:, :, 0].flatten()  # Lightness

    # Extract texture features (using LBP and Sobel filters)
    gray_image = np.mean(image, axis=2).astype(np.uint8)  # Convert to grayscale for texture analysis
    lbp = local_binary_pattern(gray_image, P=8, R=1, method="uniform").flatten()  # LBP feature
    sobel_edges = sobel(gray_image).flatten()  # Sobel edge detection
    
    # Extract spatial features (normalized X, Y coordinates)
    height, width = gray_image.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    x_normalized = (x_coords.flatten() / width).astype(np.float32)  # Normalize to [0, 1]
    y_normalized = (y_coords.flatten() / height).astype(np.float32)  # Normalize to [0, 1]
    
    # Combine all features into a single feature matrix (N x M)
    features = np.column_stack([l_channel, lbp, sobel_edges, x_normalized, y_normalized])
    
    return features

def kmeans_manual(image: np.ndarray, features, cluster_size, max_iters=200):
    image_shape = image.shape[:2]
    num_samples = features.shape[0]
    random_indices = np.random.choice(num_samples, cluster_size, replace=False)
    centroids = features[random_indices]
    
    labels = np.zeros(num_samples)
    
    for it in range(max_iters):
        # Vectorized distance calculation: Compute all distances at once
        distances = np.linalg.norm(features[:, np.newaxis] - centroids, axis=2)
        
        # Assign each pixel to the nearest centroid
        labels = np.argmin(distances, axis=1)
        
        # Recompute centroids
        new_centroids = np.array([features[labels == i].mean(axis=0) for i in range(cluster_size)])
        
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return labels.reshape(image_shape)

def kmeans_auto(image: np.ndarray, ground_truth: np.ndarray, features, max_iters=200):
    # Set a range of cluster sizes to evaluate
    cluster_sizes = range(2, 6)  # Check cluster sizes from 2 to 6
    best_score = -1
    best_cluster_size = 2
    best_labels = None

    gt_grayscale = Image.fromarray(ground_truth).convert('L')  # Convert to grayscale
    gt_grayscale = np.array(gt_grayscale)
    
    for cluster_size in cluster_sizes:
        # Perform manual K-means clustering
        labels = kmeans_manual(image, features, cluster_size, max_iters)
        
        # Calculate V-measure score for the current cluster size
        score = v_measure_score(gt_grayscale.flatten(), labels.flatten())
        
        # Keep track of the best score and corresponding cluster size
        if score > best_score:
            best_score = score
            best_cluster_size = cluster_size
            best_labels = labels
    
    return best_cluster_size, best_score, best_labels