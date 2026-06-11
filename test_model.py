from keras.models import load_model

model = load_model(
    "mask_detection_cnn_model.h5",
    compile=False
)

print("MODEL LOADED SUCCESSFULLY")
print(model.summary())