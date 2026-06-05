import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np

st.title("Real-Time Face Mask Detection")

# Load your trained model here
# model = load_model("mask_detector.model")

class FaceMaskDetector(VideoTransformerBase):

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Example rectangle
        cv2.putText(img, "Camera Working", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        return img

webrtc_streamer(
    key="mask-detection",
    video_transformer_factory=FaceMaskDetector
)