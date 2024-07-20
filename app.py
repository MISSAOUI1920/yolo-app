import streamlit as st
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# Streamlit app title
st.title("YOLO Model Deployment")


model_path = "model/best (2).pt"

# Load the model
model = YOLO(model_path)

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
        result.save("result.jpg")  # Specify filename directly

        # Load and display the saved image
        img = Image.open("result.jpg")
        st.image(img, caption='Classified Image', use_column_width=True)

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
