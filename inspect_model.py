import h5py
import json

with h5py.File("mask_detection_cnn_model.h5", "r") as f:
    config = f.attrs.get("model_config")

    if config:
        config = json.loads(config)
        print("Model class:", config.get("class_name"))

        layers = config["config"]["layers"]

        for i, layer in enumerate(layers):
            print(f"\nLayer {i}:")
            print("Class:", layer["class_name"])

            if "config" in layer:
                print(layer["config"])