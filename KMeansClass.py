import os
import numpy as np
import cv2
from PIL import Image
from skimage.color import rgb2lab
from skimage.filters import sobel
from sklearn.metrics import v_measure_score

class KMeansModel:
    def __init__(self):
        self.max_iters = None
        self.centroids = None
        self.cluster_size = None

    def extract_features(self, image: np.ndarray):
        image = cv2.fastNlMeansDenoisingColored(image, None, 30, 30, 7, 21)

        # Convert RGB to Lab color space
        lab_image = rgb2lab(image)
        l_channel = lab_image[:, :, 0].flatten()
        a_channel = lab_image[:, :, 1].flatten()
        b_channel = lab_image[:, :, 2].flatten()

        # Convert to grayscale and extract texture features
        gray_image = np.mean(image, axis=2).astype(np.uint8)
        sobel_edges = sobel(gray_image).flatten()

        # Extract spatial features (normalized coordinates)
        height, width = gray_image.shape
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        x_normalized = (x_coords.flatten() / width).astype(np.float32)
        y_normalized = (y_coords.flatten() / height).astype(np.float32)

        # Combine all features into a feature matrix (N x M)
        features = np.column_stack([l_channel, a_channel, b_channel, sobel_edges, x_normalized, y_normalized])

        return features

    def fit(self, images, cluster_size, max_iters):
        self.cluster_size = cluster_size
        self.max_iters = max_iters
        all_features = np.vstack([self.extract_features(img) for img in images])
        num_samples = all_features.shape[0]

        # Initialize centroids randomly from the dataset
        random_indices = np.random.choice(num_samples, self.cluster_size, replace=False)
        self.centroids = all_features[random_indices]

        # Iteratively update centroids
        for _ in range(self.max_iters):
            labels = self._assign_clusters(all_features)
            new_centroids = np.array([all_features[labels == i].mean(axis=0)
                                      for i in range(self.cluster_size)])
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids

    def fit_auto(self, images, masks, max_iters):
        self.max_iters = max_iters
        
        best_score = -1
        best_cluster_size = None
        best_centroids = None

        all_masks = [np.array(Image.fromarray(mask).convert('L')).flatten() for mask in masks]

        # Iterate over possible cluster sizes and train for the best score
        for cluster_size in range(2, 6):
            self.cluster_size = cluster_size
            self.fit(images, cluster_size, self.max_iters)

            all_labels = [self.predict(img).flatten() for img in images]
            score = np.mean([v_measure_score(gt, pred) for gt, pred in zip(all_masks, all_labels)])

            if score > best_score:
                best_score = score
                best_cluster_size = cluster_size
                best_centroids = self.centroids

        # Save the best model configuration
        self.cluster_size = best_cluster_size
        self.centroids = best_centroids

        return best_cluster_size

    def _assign_clusters(self, features):
        distances = np.linalg.norm(features[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def predict(self, image: np.ndarray):
        if self.centroids is None:
            raise ValueError("Model is not trained. Please train or load the model first.")
        features = self.extract_features(image)
        labels = self._assign_clusters(features)
        return labels.reshape(image.shape[:2])

    def evaluate_single(self, image: np.ndarray, mask: np.ndarray):
        predicted_labels = self.predict(image).flatten()
        mask_grayscale = np.array(Image.fromarray(mask).convert('L')).flatten()
        score = v_measure_score(mask_grayscale, predicted_labels)
        return score

    def evaluate(self, images, masks):
        all_labels = [self.predict(img).flatten() for img in images]
        all_masks = [np.array(Image.fromarray(mask).convert('L')).flatten() for mask in masks]
        scores = [v_measure_score(gt, pred) for gt, pred in zip(all_masks, all_labels)]
        return np.mean(scores)

    def save_model(self, path='kmeans_model.npz'):
        if self.centroids is None:
            raise ValueError("Model is not trained. Train the model before saving.")
        np.savez(path, centroids=self.centroids, cluster_size=self.cluster_size)

    def load_model(self, path='kmeans_model.npz'):
        data = np.load(path)
        self.centroids = data['centroids']
        self.cluster_size = int(data['cluster_size'])