import streamlit as st
import cv2
import av
import numpy as np

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide"
)

st.title("😷 AI Face Mask Detection")
st.markdown(
    "### Real-Time Face Mask Detection using TensorFlow + Streamlit"
)

# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_mask_model():
    model = load_model(
        "mask_detection.keras",
        compile=False
    )
    return model

model = load_mask_model()

# ==================================================
# LOAD FACE DETECTOR
# ==================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==================================================
# VIDEO PROCESSOR
# ==================================================

class MaskDetector(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # Convert frame to grayscale
        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        # Loop through detected faces
        for (x, y, w, h) in faces:

            try:

                face = img[y:y+h, x:x+w]

                if face.size == 0:
                    continue

                # Convert to RGB
                face_rgb = cv2.cvtColor(
                    face,
                    cv2.COLOR_BGR2RGB
                )

                # Resize to model input size
                face_rgb = cv2.resize(
                    face_rgb,
                    (224, 224)
                )

                # Preprocess image
                face_rgb = preprocess_input(
                    face_rgb.astype("float32")
                )

                # Expand dimensions
                face_rgb = np.expand_dims(
                    face_rgb,
                    axis=0
                )

                # Prediction
                prediction = model.predict(
                    face_rgb,
                    verbose=0
                )[0][0]

                # Classification
                if prediction > 0.5:

                    label = "NO MASK"
                    color = (0, 0, 255)  # Red

                else:

                    label = "MASK"
                    color = (0, 255, 0)  # Green

                # Draw rectangle
                cv2.rectangle(
                    img,
                    (x, y),
                    (x + w, y + h),
                    color,
                    3
                )

                # Put label
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

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

# ==================================================
# START WEBCAM
# ==================================================

webrtc_streamer(
    key="mask-detection",

    video_processor_factory=MaskDetector,

    rtc_configuration={
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
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