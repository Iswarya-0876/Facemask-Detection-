import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# ----------------------------
# Load Model
# ----------------------------
MODEL_PATH = "mask_detector_model.h5"   # change if your model name is different
model = tf.keras.models.load_model(MODEL_PATH)

# Labels
labels = ["Mask", "No Mask"]

# ----------------------------
# Preprocess function
# ----------------------------
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  # adjust size based on your model
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ----------------------------
# Prediction function
# ----------------------------
def predict_mask(img):
    processed = preprocess_image(img)
    prediction = model.predict(processed)[0]
    return labels[np.argmax(prediction)], prediction

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Face Mask Detection", layout="centered")

st.title("😷 Face Mask Detection App")
st.write("Upload an image or use your webcam to detect mask or no mask.")

option = st.radio("Choose input type:", ["Upload Image", "Use Webcam"])

# ----------------------------
# Upload Image
# ----------------------------
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        label, prob = predict_mask(img_array)

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {np.max(prob)*100:.2f}%")

# ----------------------------
# Webcam
# ----------------------------
elif option == "Use Webcam":
    img_file = st.camera_input("Take a picture")

    if img_file is not None:
        image = Image.open(img_file)
        img_array = np.array(image)

        st.image(image, caption="Captured Image", use_column_width=True)

        label, prob = predict_mask(img_array)

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {np.max(prob)*100:.2f}%")