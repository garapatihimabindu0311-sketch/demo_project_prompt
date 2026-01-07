import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ----------------------------
# App Configuration
# ----------------------------
st.set_page_config(
    page_title="Human Face Detection App",
    layout="centered"
)

st.title("👤 Human Face Identification")
st.write("Upload an image and the app will detect and tag human faces.")

# ----------------------------
# Load Haar Cascade
# ----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("Detection Parameters")

scale_factor = st.sidebar.slider(
    "Scale Factor (Image Reduction)",
    min_value=1.05,
    max_value=1.5,
    value=1.1,
    step=0.05
)

min_neighbors = st.sidebar.slider(
    "Min Neighbors (Detection Strictness)",
    min_value=3,
    max_value=10,
    value=5
)

# ----------------------------
# Image Upload
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Uploaded Image Preview")
    st.image(image, use_container_width=True)

    # Convert image to OpenCV format
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Face Detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(
            img_array,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
        cv2.putText(
            img_array,
            "Human face identified",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Display Result
    st.subheader("Detection Result")
    st.image(img_array, use_container_width=True)

    st.success(f"Total faces detected: {len(faces)}")

else:
    st.info("Please upload an image to start face detection.")
