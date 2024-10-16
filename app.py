import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from KMeansClass import KMeansModel
from functions import load_dataset

# Title of the app
st.title("KMeans Segmentation Demo")

# Sidebar for user inputs
st.sidebar.header("Model Configuration")
cluster_size = st.sidebar.slider("Cluster Size", min_value=2, max_value=10, value=6)
max_iters = st.sidebar.slider("Max Iterations", min_value=10, max_value=100, value=50)

# Initialize session state to keep track of images, masks, and model
if 'images' not in st.session_state:
    st.session_state.images = None
if 'masks' not in st.session_state:
    st.session_state.masks = None
if 'model' not in st.session_state:
    st.session_state.model = None  # Store the trained model here

# Load dataset and display sample images
st.subheader("Dataset")
if st.button("Load Dataset"):
    st.session_state.images, st.session_state.masks = load_dataset()
    st.success(f"Loaded {len(st.session_state.images)} images and {len(st.session_state.masks)} masks.")
    
    # Display first image and mask for visualization
    if st.session_state.images and st.session_state.masks:
        st.image([Image.fromarray(st.session_state.images[0]), Image.fromarray(st.session_state.masks[0])], caption=["Sample Image", "Sample Mask"], width=300)

# Create a KMeans model
st.subheader("Model Training")
if st.button("Train Model"):
    if st.session_state.images and st.session_state.masks:
        # Train a new KMeans model and save it to session state
        model = KMeansModel()
        model.fit(st.session_state.images, cluster_size=cluster_size, max_iters=max_iters)
        st.session_state.model = model  # Store trained model in session state
        st.success(f"Model trained with {cluster_size} clusters and {max_iters} iterations.")

# Manual image upload for prediction
st.subheader("Manual Image Upload for Prediction")
uploaded_file = st.file_uploader("Upload an image (JPG format)", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded file to a numpy array
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict segmentation on the uploaded image
    if st.session_state.model:
        predicted_labels = st.session_state.model.predict(image_np)
        
        # Display original image and predicted segmentation side by side
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image_np)
        ax[0].set_title('Uploaded Image')
        ax[1].imshow(predicted_labels, cmap="viridis")
        ax[1].set_title('Predicted Segmentation')
        st.pyplot(fig)
    else:
        st.error("Model is not trained. Please train or load a model first.")

# Evaluate the model
st.subheader("Model Evaluation")
if st.button("Evaluate Model"):
    if st.session_state.images and st.session_state.masks:
        if st.session_state.model:  # Ensure the model is trained
            score = st.session_state.model.evaluate(st.session_state.images, st.session_state.masks)
            st.write(f"Average V-measure score for the dataset: {score:.4f}")
        else:
            st.error("Model is not trained. Please train the model first.")

# Save and Load the model
st.subheader("Model Persistence")
if st.button("Save Model"):
    if st.session_state.model:  # Ensure the model is trained
        st.session_state.model.save_model("trained_kmeans_model.npz")
        st.success("Model saved as 'trained_kmeans_model.npz'.")
    else:
        st.error("Model is not trained. Please train the model first.")

if st.button("Load Model"):
    model = KMeansModel()
    model.load_model("trained_kmeans_model.npz")
    st.session_state.model = model  # Store loaded model in session state
    st.success("Model loaded from 'trained_kmeans_model.npz'.")
