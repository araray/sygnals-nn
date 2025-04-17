# tests/test_create.py
# -*- coding: utf-8 -*-
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
        activations_str="relu,sigmoid", # Hidden activation, Output activation
        loss="binary_crossentropy",
        optimizer="adam",
        output_path=str(model_path)
    )
    assert model_path.exists()
    model = tf.keras.models.load_model(model_path)
    assert len(model.layers) == 2 # Hidden layer + Output layer
    assert model.layers[0].units == 8
    assert model.layers[1].units == 1
    # Check activation names correctly
    assert 'relu' in model.layers[0].activation.__name__.lower()
    assert 'sigmoid' in model.layers[1].activation.__name__.lower()
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)

def test_create_dense_single_activation(model_path):
    """Test creating a dense network with a single activation applied to all."""
    create_network(
        layers_str="5,10,10,2", # Input=5, Hidden1=10, Hidden2=10, Output=2
        layer_types_str=None,
        activations_str="tanh", # Single activation for all 3 layers (Hidden1, Hidden2, Output)
        loss="categorical_crossentropy",
        optimizer="sgd",
        output_path=str(model_path)
    )
    assert model_path.exists()
    model = tf.keras.models.load_model(model_path)
    assert len(model.layers) == 3
    assert 'tanh' in model.layers[0].activation.__name__.lower()
    assert 'tanh' in model.layers[1].activation.__name__.lower()
    assert 'tanh' in model.layers[2].activation.__name__.lower()

# --- Test CNN Network Creation ---
def test_create_cnn1d_network_valid(model_path):
    """Test creating a simple Conv1D network."""
    create_network(
        # Fix: layers_str should match layer_types count (3 types -> 3 configs)
        layers_str="32,None,10", # Filters for conv, None for Flatten, units for dense
        layer_types_str="conv1d, flatten, dense", # Specify layer types
        # Fix: activations_str should match activatable layers (conv1d, dense -> 2 activations)
        activations_str="relu,softmax",
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
    assert 'softmax' in model.layers[2].activation.__name__.lower()

def test_create_cnn_with_pooling(model_path):
    """Test creating a CNN with MaxPooling."""
    create_network(
        # Fix: layers_str should match layer_types count (4 types -> 4 configs)
        layers_str="16,None,None,1", # Conv1D, MaxPool1D, Flatten, Dense
        layer_types_str="conv1d, maxpool1d, flatten, dense",
        # Fix: activations_str should match activatable layers (conv1d, dense -> 2 activations)
        activations_str="relu,sigmoid",
        loss="binary_crossentropy",
        optimizer="rmsprop",
        output_path=str(model_path),
        kernel_sizes_str="5", # Kernel for Conv1D
        pool_sizes_str="2", # Pool size for MaxPooling1D
        strides_str="1,2", # Stride for Conv1D, stride for MaxPooling1D
        padding_str="same", # Same padding for all layers applicable
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
    assert model.layers[1].padding == 'same' # Keras applies padding from Conv here if 'same'
    assert isinstance(model.layers[2], tf.keras.layers.Flatten)
    assert isinstance(model.layers[3], tf.keras.layers.Dense)
    assert model.layers[3].units == 1
    assert 'sigmoid' in model.layers[3].activation.__name__.lower()

# --- Test Invalid Configurations ---
def test_create_network_mismatch_activations(model_path):
    """Test error when activation count doesn't match activatable layer count."""
    # layers_str="4,8,1" -> Input Dim 4, Dense(8), Dense(1) -> 2 activatable layers
    # activations_str="relu,sigmoid,tanh" -> 3 activations
    with pytest.raises(ValueError, match="Mismatch: Provided 3 activations .* but expected 2"):
        create_network(
            layers_str="4,8,1",
            output_path=str(model_path),
            layer_types_str=None, # Defaults to dense
            activations_str="relu,sigmoid,tanh", # Too many
            loss="mse",
            optimizer="adam"
        )

def test_create_network_mismatch_layer_types(model_path):
    """Test error when layer type count doesn't match layer config count."""
    # layers_str="4,8,16,1" -> Input 4, Dense(8), Dense(16), Dense(1) -> 3 layers to build
    # layer_types_str="dense,dense" -> 2 types specified
    with pytest.raises(ValueError, match="Mismatch: 3 layer configurations .* but 2 layer types specified"):
         create_network(
             layers_str="4,8,16,1",
             output_path=str(model_path),
             layer_types_str="dense,dense", # Too few
             activations_str="relu,relu,sigmoid", # Matches 3 layers
             loss="mse",
             optimizer="adam"
        )

def test_create_cnn_missing_input_shape(model_path):
    """Test error when creating CNN without specifying input shape."""
    # Fix: Adjust match string to the actual error raised later in the function
    with pytest.raises(ValueError, match='Input shape/dimension could not be determined for the first layer.'):
        create_network(
            layers_str="None,32,1", # Config implies Conv start
            output_path=str(model_path),
            layer_types_str="conv1d, flatten, dense",
            activations_str="relu,sigmoid", # Matches activatable layers
            loss="mse",
            optimizer="adam",
            input_shape_str=None # Missing
        )

def test_create_unsupported_layer_type(model_path):
    """Test error with an unknown layer type."""
    with pytest.raises(ValueError, match="Unsupported layer type: 'lstm'"):
         create_network(
             # Fix: Adjust layers_str to match layer_types_str length (2 types -> 2 configs)
             layers_str="20,1", # Config for LSTM (20 units), Dense (1 unit)
             output_path=str(model_path),
             layer_types_str="lstm, dense", # Unsupported type 'lstm'
             activations_str="tanh, sigmoid", # Matches activatable layers
             loss="mse",
             optimizer="adam",
             input_shape_str="5,10" # Need input shape for LSTM
        )
