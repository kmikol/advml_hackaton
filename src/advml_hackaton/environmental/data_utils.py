import numpy as np
import yaml
import os

def get_data(data_path):


    train = np.load(os.path.join(data_path, "processed", "train.npz"))
    val = np.load(os.path.join(data_path, "processed", "val.npz"))
    test = np.load(os.path.join(data_path, "processed", "test.npz"))

    # load from yaml
    data_params = yaml.safe_load(open(os.path.join(data_path, "processed", "data_meta.yaml")))

    return train, val, test, data_params

def get_data_meta(data_path):
    return yaml.safe_load(open(os.path.join(data_path, "processed", "data_meta.yaml"))) 
