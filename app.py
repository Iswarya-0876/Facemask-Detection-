import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import av
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# PAGE SETTINGS
# =========================

st.set_page_config(
    page_title="Face Mask Detection",
    layout="centered"
)

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

        # Frame skipping for smooth video
        self.frame_count += 1

        if self.frame_count % 2 != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60)
        )

        # =========================
        # LOOP THROUGH FACES
        # =========================

        for (x, y, w, h) in faces:

            try:
                # Extract face
                face = img[y:y+h, x:x+w]

                # Convert BGR to RGB
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                # Resize for model
                face = cv2.resize(face, (224, 224))

                # Convert to float32
                face = np.array(face, dtype="float32")

                # Preprocess for MobileNetV2
                face = preprocess_input(face)

                # Expand dimensions
                face = np.expand_dims(face, axis=0)

                # Prediction
                prediction = model.predict(face, verbose=0)[0][0]

                print("Prediction:", prediction)

                # =========================
                # LABEL LOGIC
                # =========================

                if prediction > 0.5:
                    label = "NO MASK"
                    color = (0, 0, 255)

                else:
                    label = "MASK"
                    color = (0, 255, 0)

                # =========================
                # DRAW RECTANGLE
                # =========================

                cv2.rectangle(
                    img,
                    (x, y),
                    (x + w, y + h),
                    color,
                    3
                )

                # Put text
                cv2.putText(
                    img,
                    f"{label} ({prediction:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

            except Exception as e:
                print("Error:", e)

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