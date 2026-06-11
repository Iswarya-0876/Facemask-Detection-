from keras.models import load_model

model = load_model(
    "mask_detection.keras",
    compile=False
)