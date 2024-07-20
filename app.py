import streamlit as st
import requests
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import io

# Streamlit app title
st.title("YOLO Model Deployment")

# GitHub repository URL for the model
github_repo_url = "https://github.com/MISSAOUI1920/yolo-app/blob/main/model/best%20(2).pt"  # Replace with your GitHub URL

# Download the model file from GitHub
response = requests.get(github_repo_url)
if response.status_code == 200:
    model_bytes = io.BytesIO(response.content)
else:
    st.error("Failed to download model file.")
    st.stop()

# Load the model
model = YOLO(model_bytes)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Perform inference
    results = model([image])  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs

        # Save the result to a file
        result_img = result.save()  # Save result to a file-like object
        st.image(result_img, caption='Classified Image', use_column_width=True)

        # Optional: Display individual components if needed
        if boxes is not None:
            st.write("Bounding Boxes:")
            st.write(boxes)
        if masks is not None:
            st.write("Segmentation Masks:")
            st.write(masks)
        if keypoints is not None:
            st.write("Keypoints:")
            st.write(keypoints)
        if probs is not None:
            st.write("Classification Probabilities:")
            st.write(probs)
        if obb is not None:
            st.write("Oriented Bounding Boxes:")
            st.write(obb)
