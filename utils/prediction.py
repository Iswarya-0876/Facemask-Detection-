import numpy as np

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from utils.config import IMG_SIZE


def preprocess_face(face):

    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

    face = preprocess_input(face)

    face = np.expand_dims(face, axis=0)

    return face