import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("mask_detection.keras")

img = cv2.imread("test_mask.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224,224))

img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)

print(pred)