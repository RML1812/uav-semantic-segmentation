import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from KMeansClass import KMeansModel
from functions import load_dataset

# Title of the app
st.title("KMeans Segmentation Demo (Manual or Auto)")

# Sidebar for user inputs
st.sidebar.header("Model Configuration")
mode = st.sidebar.radio("Cluster Size Mode", ["Manual", "Automatic"])
max_iters = st.sidebar.slider("Max Iterations", min_value=10, max_value=100, value=50)

# If in manual mode, ask for cluster size
if mode == "Manual":
    cluster_size = st.sidebar.slider("Cluster Size", min_value=2, max_value=10, value=6)

# Initialize session state to keep track of images, masks, and model
if 'images' not in st.session_state:
    st.session_state.images = None
if 'masks' not in st.session_state:
    st.session_state.masks = None
if 'model' not in st.session_state:
    st.session_state.model = None  # Store the trained model here
if 'cluster_size' not in st.session_state:
    st.session_state.cluster_size = None  # Store the optimal cluster size found by fit_auto()

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
        # Initialize the KMeans model
        model = KMeansModel()

        # Train based on the user's selected mode
        if mode == "Manual":
            # Manual mode: Train the model with a user-specified cluster size
            model.fit(st.session_state.images, cluster_size=cluster_size, max_iters=max_iters)
            st.session_state.model = model  # Store trained model in session state
            st.session_state.cluster_size = cluster_size  # Store the manually selected cluster size
            st.success(f"Model trained manually with {cluster_size} clusters and {max_iters} iterations.")
        else:
            # Automatic mode: Use fit_auto to determine the best cluster size
            optimal_cluster_size = model.fit_auto(st.session_state.images, st.session_state.masks, max_iters=max_iters)
            st.session_state.model = model  # Store trained model in session state
            st.session_state.cluster_size = optimal_cluster_size  # Store the optimal cluster size
            st.success(f"Model trained automatically with {optimal_cluster_size} optimal clusters and {max_iters} iterations.")

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

# Evaluation options: single image or entire dataset
st.subheader("Model Evaluation")
eval_option = st.selectbox("Choose evaluation type", ["Evaluate Single Image", "Evaluate Full Dataset"])

# Evaluate single image
if eval_option == "Evaluate Single Image":
    st.subheader("Evaluate Single Image")
    eval_uploaded_file = st.file_uploader("Upload an image with ground truth mask (JPG format)", type=["jpg", "jpeg"], key="eval_upload")
    mask_uploaded_file = st.file_uploader("Upload the corresponding ground truth mask (JPG format)", type=["jpg", "jpeg"], key="mask_upload")
    
    if eval_uploaded_file is not None and mask_uploaded_file is not None:
        eval_image = Image.open(eval_uploaded_file)
        eval_image_np = np.array(eval_image)

        mask_image = Image.open(mask_uploaded_file)
        mask_image_np = np.array(mask_image)

        # Evaluate single image using the model
        if st.session_state.model:
            score = st.session_state.model.evaluate_single(eval_image_np, mask_image_np)
            st.write(f"V-measure score for the uploaded image: {score:.4f}")
        else:
            st.error("Model is not trained. Please train or load the model first.")

# Evaluate full dataset
if eval_option == "Evaluate Full Dataset":
    st.subheader("Evaluate Full Dataset")
    if st.button("Evaluate Entire Dataset"):
        if st.session_state.images and st.session_state.masks:
            if st.session_state.model:  # Ensure the model is trained
                score = st.session_state.model.evaluate(st.session_state.images, st.session_state.masks)
                st.write(f"Average V-measure score for the dataset: {score:.4f}")
            else:
                st.error("Model is not trained. Please train the model first.")
        else:
            st.error("Dataset is not loaded. Please load the dataset first.")

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
