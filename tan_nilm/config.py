import os

CONFIG = {
    "dataset_path": os.path.join(os.getcwd(), "dataset/ukdale/ukdale_tan.h5"),
    "houses": [1, 2, 5],  # Houses to train/test
    "appliances": {1: "fridge", 10: "washing_machine"},
    "sequence_length": 512,
    "batch_size": 64,
    "epochs": 20,
    "learning_rate": 0.001,
    "model_save_path": os.path.join(os.getcwd(), "tan_nilm_model.pth")
}
