import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av

st.title("Face Mask Detection")

webrtc_streamer(key="camera")