import pytest
import os
import tensorflow as tf
from sygnals_nn.create import create_network

def test_create_network_valid():
    output_file = "test_model.keras"
    if os.path.exists(output_file):
        os.remove(output_file)
    create_network("4,8,1", "relu,sigmoid", "binary_crossentropy", "adam", output_file)
    assert os.path.exists(output_file)

    model = tf.keras.models.load_model(output_file)
    # Instead of checking output_shape, check number of units in first layer
    assert len(model.layers) == 2
    assert model.layers[0].units == 8

def test_create_network_mismatch_activations():
    with pytest.raises(ValueError):
        # 3 activations for a 2-layer network should raise an error
        create_network("4,8,1", "relu,sigmoid,tanh", "binary_crossentropy", "adam", "dummy_model.keras")
