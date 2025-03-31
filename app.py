import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
from ultralytics import YOLO
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# ---- CONFIGURATION ----
# Paths to the locally downloaded models
CNN_MODEL_PATH = "C:\\Phase 2\\Brain tumor\\BrainTumorApp\\brain_tumor_model_relu (1).h5"
YOLO_MODEL_PATH = "C:\\Phase 2\\Brain tumor\\BrainTumorApp\\yolov8_tumor.pt"

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# ---- LOAD MODELS ----
cnn_model = load_model(CNN_MODEL_PATH)
yolo_model = YOLO(YOLO_MODEL_PATH)

# ---- FUNCTION TO PREPROCESS IMAGE ----
def preprocess_image(image):
    """Preprocess the image for CNN model"""
    img = image.resize((224, 224))  # Resize to CNN input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# ---- FUNCTION FOR STAGE 1: CLASSIFICATION ----
def classify_image(image):
    """Classify the image using CNN model"""
    processed_image = preprocess_image(image)
    predictions = cnn_model.predict(processed_image)
    class_index = np.argmax(predictions)
    class_name = CLASS_NAMES[class_index]
    confidence = np.max(predictions)
    return class_name, confidence

# ---- FUNCTION FOR STAGE 2: TUMOR DETECTION ----
def detect_tumor(image):
    """Detect tumor using YOLO model"""
    img_array = np.array(image)
    results = yolo_model(img_array)  # Run YOLO model

    for result in results:
        if len(result.boxes) > 0:  # If a tumor is detected
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = float(box.conf[0])  # Confidence score

                # Draw bounding box
                cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img_array, f"Tumor: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return Image.fromarray(img_array)  # Convert back to PIL Image

# ---- STREAMLIT INTERFACE ----
st.title("🧠 Brain Tumor Classification & Detection")
st.write("Upload an MRI image to classify and detect brain tumors.")

# **Step 1: Upload Image**
uploaded_file = st.file_uploader("Choose an MRI Image...", type=["jpg", "png", "jpeg"])

# **Reset session state when a new image is uploaded**
if uploaded_file is not None:
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        st.session_state.classification_result = None  # Reset classification result
        st.session_state.classification_confidence = None
        st.session_state.classification_done = False  # Reset classification status
        st.session_state.last_uploaded = uploaded_file.name  # Store new image name

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # **Step 2: Classification Button**
    if not st.session_state.classification_done:
        if st.button("Classify Tumor"):
            with st.spinner("Classifying..."):
                class_name, confidence = classify_image(image)

            st.success(f"Prediction: **{class_name}** with confidence **{confidence:.2f}**")

            # Store classification result in session state
            st.session_state.classification_result = class_name
            st.session_state.classification_confidence = confidence
            st.session_state.classification_done = True

    # **Step 3: Tumor Detection Button (Only if Tumor is detected)**
    if st.session_state.classification_done and st.session_state.classification_result != "No Tumor":
        if st.button("Proceed to Tumor Detection"):
            with st.spinner("Detecting tumor region..."):
                detected_image = detect_tumor(image)

            st.image(detected_image, caption="Tumor Detection Result", use_column_width=True)
            st.success("Tumor detection complete!")
