import streamlit as st
import cv2
import av
import numpy as np

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase
)

from tensorflow.keras.models import load_model


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="AI Face Mask Detection",
    page_icon="😷",
    layout="wide"
)

st.title("😷 AI Face Mask Detection")
st.write("Real-Time Face Mask Detection")


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_mask_model():
    return load_model(
        "mask_detection.keras",
        compile=False
    )

model = load_mask_model()


# ==================================================
# FACE DETECTOR
# ==================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    st.error("Failed to load Haar Cascade")
    st.stop()


# ==================================================
# VIDEO PROCESSOR
# ==================================================

class MaskDetector(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces:

            face = img[y:y+h, x:x+w]

            if face.size == 0:
                continue

            try:

                # Convert BGR → RGB
                face_rgb = cv2.cvtColor(
                    face,
                    cv2.COLOR_BGR2RGB
                )

                # Resize to model input size
                face_rgb = cv2.resize(
                    face_rgb,
                    (224, 224)
                )

                # NORMALIZATION USED IN CUSTOM CNN
                face_rgb = face_rgb.astype(
                    "float32"
                ) / 255.0

                face_rgb = np.expand_dims(
                    face_rgb,
                    axis=0
                )

                prediction = model.predict(
                    face_rgb,
                    verbose=0
                )[0][0]

                print("Prediction:", prediction)

                # -------------------------------------------------
                # LABEL MAPPING
                # -------------------------------------------------

                if prediction > 0.5:

                    label = "NO MASK"

                    color = (0, 0, 255)



                else:

                    label = "MASK"

                    color = (0, 255, 0)


                cv2.rectangle(
                    img,
                    (x, y),
                    (x + w, y + h),
                    color,
                    3
                )

                cv2.putText(
                    img,
                    f"{label} {confidence:.1f}%",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

            except Exception as e:
                print(e)

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# ==================================================
# WEBCAM STREAM
# ==================================================

webrtc_streamer(
    key="mask-detection",

    video_processor_factory=MaskDetector,

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    async_processing=True
)