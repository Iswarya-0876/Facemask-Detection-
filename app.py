import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import av
from tensorflow.keras.models import load_model

# =========================
# PAGE SETTINGS
# =========================

st.set_page_config(page_title="Face Mask Detection", layout="centered")

st.title("😷 Live Face Mask Detection")
st.write("Real-time AI webcam detection using Streamlit + TensorFlow")

# =========================
# LOAD MODEL
# =========================

model = load_model("mask_detection_cnn_model.h5")

# =========================
# LOAD FACE DETECTOR
# =========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# VIDEO PROCESSOR CLASS
# =========================

class MaskDetector(VideoProcessorBase):

    def __init__(self):
        self.frame_count = 0

    def recv(self, frame):

        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")

        # Frame skipping for smooth live video
        self.frame_count += 1

        if self.frame_count % 3 != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        # Loop through detected faces
        for (x, y, w, h) in faces:

            face = img[y:y+h, x:x+w]

            try:
                # Resize face for model
                face_resized = cv2.resize(face, (224, 224))

                # Normalize
                face_resized = face_resized.astype("float32") / 255.0

                # Expand dimensions
                face_resized = np.expand_dims(face_resized, axis=0)

                # Prediction
                prediction = model(face_resized, training=False).numpy()

                confidence = prediction[0][0]

                # =========================
                # MASK DETECTION LOGIC
                # =========================

                if confidence > 0.5:
                    label = "MASK"
                    color = (0, 255, 0)   # Green
                else:
                    label = "NO MASK"
                    color = (0, 0, 255)   # Red

                # Draw rectangle
                cv2.rectangle(
                    img,
                    (x, y),
                    (x + w, y + h),
                    color,
                    3
                )

                # Put label text
                cv2.putText(
                    img,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2
                )

            except Exception as e:
                print(e)

        # Return processed frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =========================
# START LIVE CAMERA
# =========================

webrtc_streamer(
    key="mask-detection",

    video_processor_factory=MaskDetector,

    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },

    media_stream_constraints={
        "video": {
            "width": 640,
            "height": 480
        },
        "audio": False,
    },

    async_processing=True,
)