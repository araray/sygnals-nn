import pytest
import os
import pandas as pd
from sygnals_nn.train import train_model
from sygnals_nn.create import create_network
import tensorflow as tf

def test_train_model():
    # Create a tiny dataset
    data = """f1,f2,label
    0,0,0
    0,1,1
    1,0,1
    1,1,0
    """
    data_file = "train_data.csv"
    with open(data_file, 'w') as f:
        f.write(data)

    model_file = "test_train_model.keras"
    if os.path.exists(model_file):
        os.remove(model_file)

    # Create a simple model
    create_network("2,2,1", "relu,sigmoid", "binary_crossentropy", "adam", model_file)

    # Train the model
    train_model(model_file, data_file, epochs=1, batch_size=1, learning_rate=0.01)

    # Check if model still loads
    model = tf.keras.models.load_model(model_file)
    assert model is not None
