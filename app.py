import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# Streamlit app title
st.title("YOLO Model Deployment")

# Define the model directory and path
model_directory = "model"
model_filename = "best.pt"  # Adjust if needed
model_path = os.path.join(model_directory, model_filename)

# Check if the model file exists
if not os.path.isfile(model_path):
    st.error(f"Model file not found: {model_path}")
else:
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
            # Display bounding boxes
            if result.boxes:
                st.write("Bounding Boxes:")
                st.write(result.boxes)

            # Display segmentation masks
            if result.masks:
                st.write("Segmentation Masks:")
                st.write(result.masks)

            # Display keypoints
            if result.keypoints:
                st.write("Keypoints:")
                st.write(result.keypoints)

            # Display classification probabilities
            if result.probs:
                st.write("Classification Probabilities:")
                st.write(result.probs)

            # Display oriented bounding boxes
            if result.obb:
                st.write("Oriented Bounding Boxes:")
                st.write(result.obb)

            # Save and display the result image
            result_image_path = os.path.join("result.jpg")  # Specify the result image filename
            result.save(result_image_path)
            result_image = Image.open(result_image_path)
            st.image(result_image, caption='Classified Image', use_column_width=True)
