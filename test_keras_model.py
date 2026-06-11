from keras.models import load_model

model = load_model(
    "mask_detection.keras",
    compile=False
)

print("MODEL LOADED SUCCESSFULLY")
print(model.summary())