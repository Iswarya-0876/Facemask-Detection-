import h5py

with h5py.File("mask_detection_cnn_model.h5", "r") as f:

    weights = f["model_weights"]

    print("Layers found:\n")

    for layer in weights.keys():
        print(layer)