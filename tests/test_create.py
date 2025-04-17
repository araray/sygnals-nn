import pytest
import os
import tensorflow as tf
from sygnals_nn.create import create_network
import logging

# Ensure logs are captured during testing
logging.basicConfig(level=logging.INFO)

# Fixture to manage temporary file creation and cleanup
@pytest.fixture
def model_path(tmp_path):
    """Provides a temporary path for saving the model."""
    return tmp_path / "test_model.keras"

# --- Test Dense Network Creation ---
def test_create_dense_network_valid(model_path):
    """Test creating a standard dense network."""
    create_network(
        layers_str="4,8,1", # Input=4, Hidden=8, Output=1
        layer_types_str=None, # Default to dense
        activations_str="relu,sigmoid",
        loss="binary_crossentropy",
        optimizer="adam",
        output_path=str(model_path)
    )
    assert model_path.exists()
    model = tf.keras.models.load_model(model_path)
    assert len(model.layers) == 2 # Hidden layer + Output layer
    assert model.layers[0].units == 8
    assert model.layers[1].units == 1
    assert 'relu' in model.layers[0].activation.__name__
    assert 'sigmoid' in model.layers[1].activation.__name__
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
    # Loss function check is less direct after loading, check config if needed
    # assert model.loss == "binary_crossentropy" # This doesn't work directly

def test_create_dense_single_activation(model_path):
    """Test creating a dense network with a single activation applied to all."""
    create_network(
        layers_str="5,10,10,2",
        layer_types_str=None,
        activations_str="tanh", # Single activation
        loss="categorical_crossentropy",
        optimizer="sgd",
        output_path=str(model_path)
    )
    assert model_path.exists()
    model = tf.keras.models.load_model(model_path)
    assert len(model.layers) == 3
    assert 'tanh' in model.layers[0].activation.__name__
    assert 'tanh' in model.layers[1].activation.__name__
    assert 'tanh' in model.layers[2].activation.__name__ # Last layer also gets tanh here

# --- Test CNN Network Creation ---
def test_create_cnn1d_network_valid(model_path):
    """Test creating a simple Conv1D network."""
    create_network(
        layers_str="None,32,64,10", # Filters for conv, units for dense
        layer_types_str="conv1d, flatten, dense", # Specify layer types
        activations_str="relu, relu, softmax", # Activations for Conv and Dense
        loss="categorical_crossentropy",
        optimizer="adam",
        output_path=str(model_path),
        kernel_sizes_str="3", # Kernel size for Conv1D
        input_shape_str="50,5" # Example: 50 timesteps, 5 features
    )
    assert model_path.exists()
    model = tf.keras.models.load_model(model_path)
    assert len(model.layers) == 3
    assert isinstance(model.layers[0], tf.keras.layers.Conv1D)
    assert model.layers[0].filters == 32
    assert model.layers[0].kernel_size == (3,)
    assert isinstance(model.layers[1], tf.keras.layers.Flatten)
    assert isinstance(model.layers[2], tf.keras.layers.Dense)
    assert model.layers[2].units == 10
    assert 'softmax' in model.layers[2].activation.__name__

def test_create_cnn_with_pooling(model_path):
    """Test creating a CNN with MaxPooling."""
    create_network(
        layers_str="None,16,8,1", # Filters, filters, dense units
        layer_types_str="conv1d, maxpool1d, flatten, dense",
        activations_str="relu, linear, sigmoid", # Pool has no activation, specify for others
        loss="binary_crossentropy",
        optimizer="rmsprop",
        output_path=str(model_path),
        kernel_sizes_str="5,3", # Kernel for Conv1D (pool doesn't use kernel_size arg this way)
        pool_sizes_str="2", # Pool size for MaxPooling1D
        strides_str="1,2", # Stride for Conv1D, stride for MaxPooling1D
        padding_str="same", # Same padding for all layers
        input_shape_str="100,3" # 100 timesteps, 3 features
    )
    assert model_path.exists()
    model = tf.keras.models.load_model(model_path)
    assert len(model.layers) == 4
    assert isinstance(model.layers[0], tf.keras.layers.Conv1D)
    assert model.layers[0].filters == 16
    assert model.layers[0].padding == 'same'
    assert isinstance(model.layers[1], tf.keras.layers.MaxPooling1D)
    assert model.layers[1].pool_size == (2,)
    assert model.layers[1].strides == (2,)
    assert model.layers[1].padding == 'same'
    assert isinstance(model.layers[2], tf.keras.layers.Flatten)
    assert isinstance(model.layers[3], tf.keras.layers.Dense)
    assert model.layers[3].units == 1

# --- Test Invalid Configurations ---
def test_create_network_mismatch_activations(model_path):
    """Test error when activation count doesn't match layer count."""
    with pytest.raises(ValueError, match="Mismatch: 2 layers require activation, but 3 provided"):
        create_network("4,8,1", None, "relu,sigmoid,tanh", "mse", "adam", str(model_path))

def test_create_network_mismatch_layer_types(model_path):
    """Test error when layer type count doesn't match layer config count."""
    with pytest.raises(ValueError, match="Mismatch: 4 layer sizes/filters provided.*but 2 layer types specified"):
         create_network("4,8,16,1", "dense,dense", "relu,relu,sigmoid", "mse", "adam", str(model_path))

def test_create_cnn_missing_input_shape(model_path):
    """Test error when creating CNN without specifying input shape."""
    with pytest.raises(ValueError, match="Must specify --input-shape for models starting with Conv layers"):
        create_network(
            layers_str="None,32,1",
            layer_types_str="conv1d, flatten, dense",
            activations_str="relu,relu,sigmoid",
            loss="mse", optimizer="adam", output_path=str(model_path)
        )

def test_create_unsupported_layer_type(model_path):
    """Test error with an unknown layer type."""
    with pytest.raises(ValueError, match="Unsupported layer type: 'lstm'"):
         create_network("10,20,1", "lstm, dense", "tanh, sigmoid", "mse", "adam", str(model_path), input_shape_str="5,10")
