# UAV Semantic Segmentation Clustering

## Contributors

| Name                                     | NPM            |
|------------------------------------------|----------------|
| [Muhammad Giat](https://github.com/mhmmadgiatt) | 140810210013 |
| [Raditya Muhamad Lacavi](https://github.com/RML1812) | 140810210019 |
| [Varian Avila Faldi](https://github.com/Varianrian) | 140810210053 |

---

## Description

This project focuses on **semantic segmentation of UAV (Unmanned Aerial Vehicle) imagery** using **KMeans clustering** implemented in Python. It provides an **interactive web-based interface** powered by **Streamlit**, allowing users to explore, train, and visualize the segmentation results directly from their browser. Users can either manually define cluster sizes or allow the app to automatically determine the optimal clusters for segmentation.

---

## Dataset

The aerial imagery dataset used for this project can be found on Kaggle:
[Semantic Segmentation of Aerial Imagery](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery)

---

## Functionality

The app offers the following features:

1. **Cluster Size Selection**:
   - Manual mode: Users manually specify the number of clusters.
   - Automatic mode: The app determines the optimal number of clusters using the `fit_auto()` method.

2. **Dataset Loading**:
   - Load UAV images and their corresponding masks from the provided dataset.
   - Display sample images and masks for quick visualization.

3. **Model Training**:
   - Train the KMeans model using either manual or automatic cluster size selection.
   - Store the trained model for future predictions or evaluations.

4. **Segmentation Prediction**:
   - Perform segmentation on:
     - **Sample images** from the loaded dataset.
     - **Uploaded images** manually provided by the user.
   - Display the original image, predicted segmentation, and ground truth mask side-by-side.

5. **Model Evaluation**:
   - Evaluate the segmentation on a **single uploaded image** with a ground truth mask.
   - Compute the **V-measure score** to assess clustering quality.
   - Option to evaluate the **entire dataset** to get an average V-measure score.

6. **Model Persistence**:
   - **Save** the trained model to disk for reuse.
   - **Load** a previously saved model for further predictions and evaluation.

---

## Installation and Running the App

### Prerequisites

- Make sure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).

### Step 1: Clone the Repository

```bash
git clone https://github.com/uav-semantic-segmentation-clustering.git
cd uav-semantic-segmentation-clustering
```

### Step 2: Install Dependencies

Use the following command to install the required libraries from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit App

Execute the following command to start the Streamlit application:

```bash
streamlit run app.py
```

### Step 4: Access the App

After running the above command, the terminal will display a **local URL** (e.g., `http://localhost:8501`). Open it in your browser to access the app.

---

## How to Use the App

1. **Load Dataset**: Click the **"Load Dataset"** button to load sample images and masks.
2. **Configure Model**: Choose manual or automatic mode for cluster size selection.
3. **Train Model**: Click **"Train Model"** to fit the KMeans model on the dataset.
4. **Prediction & Evaluation**: Use provided buttons to test the model on sample or uploaded images.
5. **Save/Load Model**: Save the trained model for future use or load an existing model.

---

## License

This project is open-source and available under the [MIT License](LICENSE).