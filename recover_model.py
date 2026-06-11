from keras.models import Sequential
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense
)

model = Sequential([
    Input(shape=(224,224,3)),

    Conv2D(32,(3,3),activation="relu"),
    MaxPooling2D((2,2)),

    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D((2,2)),

    Conv2D(128,(3,3),activation="relu"),
    MaxPooling2D((2,2)),

    Flatten(),

    Dense(128,activation="relu"),

    Dense(1,activation="sigmoid")
])

model.build((None,224,224,3))

print("Model built")

model.load_weights("mask_detection_cnn_model.h5")

print("Weights loaded")

model.save("mask_detection.keras")

print("Model saved successfully")