# tests/test_train.py
# -*- coding: utf-8 -*-
import pytest
import os
import pandas as pd
import numpy as np
from sygnals_nn.train import train_model
from sygnals_nn.create import create_network
import json
import traceback # FIX: Added missing import

# Force TensorFlow to use CPU only to avoid CUDA/CuDNN issues in test environments
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Import tensorflow after setting the environment variable
import tensorflow as tf


# Fixture to create necessary files for training tests
@pytest.fixture
def train_setup(tmp_path):
    """Creates model and data files for training tests."""
    setup_files = {}

    # --- Create Basic CSV Data ---
    csv_train_path = tmp_path / "train_train.csv"
    pd.DataFrame({
        'col_a': np.random.rand(20),
        'col_b': np.random.rand(20),
        'target': np.random.randint(0, 2, size=20)
    }).to_csv(csv_train_path, index=False)
    setup_files["csv_train"] = csv_train_path

    # --- Create Basic JSON Data ---
    json_train_path = tmp_path / "train_train.json"
    json_data = [
        {'features': [0.1, 0.9], 'label': 0},
        {'features': [0.2, 0.8], 'label': 1},
        {'features': [0.3, 0.7], 'label': 0},
        {'features': [0.4, 0.6], 'label': 1},
        {'features': [0.5, 0.5], 'label': 0},
    ]
    with open(json_train_path, 'w') as f:
        json.dump(json_data, f)
    setup_files["json_train"] = json_train_path

    # --- Create Basic Keras Model (2 inputs, 1 output) ---
    keras_model_path = tmp_path / "train_model.keras"
    # Create the initial model structure, training will load and modify it
    # Corrected call using keyword arguments
    create_network(
        layers_str="2,4,1",           # Input dim 2, hidden 4, output 1
        output_path=str(keras_model_path),
        layer_types_str=None,         # Default to dense
        activations_str="relu,sigmoid", # Activation for hidden, output
        loss="binary_crossentropy",
        optimizer="adam"
    )
    setup_files["keras_model"] = keras_model_path

    # --- Path for ONNX export ---
    setup_files["onnx_export"] = tmp_path / "trained_model.onnx"

    return setup_files

# --- Test Basic Training ---
def test_train_model_csv_success(train_setup):
    """Test successful training using CSV data."""
    model_file = train_setup["keras_model"]
    data_file = train_setup["csv_train"]

    # Get initial weights to check they change after training
    model_before = tf.keras.models.load_model(model_file)
    weights_before = [w.numpy() for w in model_before.weights]

    train_model(
        model_path=str(model_file),
        data_path=str(data_file),
        epochs=2,
        batch_size=4,
        learning_rate=0.01,
        input_cols_str='col_a,col_b', # Use names
        label_cols_str='target'
    )

    # Check if model file still exists and load it
    assert model_file.exists()
    model_after = tf.keras.models.load_model(model_file)
    weights_after = [w.numpy() for w in model_after.weights]

    # Check that weights have changed
    assert len(weights_before) == len(weights_after)
    weight_changed = False
    for w_before, w_after in zip(weights_before, weights_after):
        if not np.array_equal(w_before, w_after):
            weight_changed = True
            break
    assert weight_changed, "Model weights did not change after training."

def test_train_model_json_success(train_setup):
    """Test successful training using JSON data."""
    model_file = train_setup["keras_model"]
    data_file = train_setup["json_train"]

    model_before = tf.keras.models.load_model(model_file)
    weights_before = [w.numpy() for w in model_before.weights]

    # Run the train function and capture potential exceptions for debugging
    try:
        train_model(
            model_path=str(model_file),
            data_path=str(data_file),
            epochs=2,
            batch_size=1,
            learning_rate=0.01,
            input_cols_str='features', # Key containing feature list
            label_cols_str='label',    # Key containing label
            json_input_key='features', # Specify JSON key for input
            json_label_key='label'     # Specify JSON key for label
        )
    except Exception as e:
        # FIX: Use imported traceback
        pytest.fail(f"train_model raised an exception: {e}\n{traceback.format_exc()}")


    assert model_file.exists()
    model_after = tf.keras.models.load_model(model_file)
    weights_after = [w.numpy() for w in model_after.weights]

    # Check weights changed
    weight_changed = False
    for w_before, w_after in zip(weights_before, weights_after):
        if not np.array_equal(w_before, w_after):
            weight_changed = True
            break
    assert weight_changed, "Model weights did not change after training (JSON)."


# --- Test Training with ONNX Export ---
def test_train_model_with_onnx_export(train_setup):
    """Test training with the --export-onnx flag."""
    # Skip if tf2onnx is not installed
    pytest.importorskip("tf2onnx")

    model_file = train_setup["keras_model"]
    data_file = train_setup["csv_train"]
    onnx_file = train_setup["onnx_export"]

    train_model(
        model_path=str(model_file),
        data_path=str(data_file),
        epochs=1,
        batch_size=4,
        learning_rate=0.01,
        input_cols_str='0,1', # Use indices
        label_cols_str='2',
        export_onnx_path=str(onnx_file) # Enable export
    )

    assert model_file.exists() # Keras model should still be saved
    assert onnx_file.exists() # ONNX model should be created

# --- Test Error Handling ---
def test_train_model_file_not_found(train_setup):
    """Test error when Keras model file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        train_model(
            model_path="non_existent_model.keras",
            data_path=str(train_setup["csv_train"]),
            epochs=1, batch_size=1, learning_rate=0.01,
            input_cols_str='0,1', label_cols_str='2'
        )

def test_train_data_file_not_found(train_setup):
    """Test error when data file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        train_model(
            model_path=str(train_setup["keras_model"]),
            data_path="non_existent_data.csv",
            epochs=1, batch_size=1, learning_rate=0.01,
            input_cols_str='0,1', label_cols_str='2'
        )

def test_train_missing_input_cols(train_setup):
    """Test error when input columns are not specified."""
    with pytest.raises(ValueError, match="Input columns must be specified"):
        train_model(
            model_path=str(train_setup["keras_model"]),
            data_path=str(train_setup["csv_train"]),
            epochs=1, batch_size=1, learning_rate=0.01,
            input_cols_str=None, # Missing
            label_cols_str='2'
        )

def test_train_missing_label_cols(train_setup):
    """Test error when label columns are not specified."""
    with pytest.raises(ValueError, match="Label columns must be specified"):
        train_model(
            model_path=str(train_setup["keras_model"]),
            data_path=str(train_setup["csv_train"]),
            epochs=1, batch_size=1, learning_rate=0.01,
            input_cols_str='0,1',
            label_cols_str=None # Missing
        )

def test_train_column_not_found(train_setup):
    """Test error when specified column name doesn't exist."""
    with pytest.raises(ValueError, match="Column name 'non_existent_col' not found"):
        train_model(
            model_path=str(train_setup["keras_model"]),
            data_path=str(train_setup["csv_train"]),
            epochs=1, batch_size=1, learning_rate=0.01,
            input_cols_str='col_a,non_existent_col', # Bad column name
            label_cols_str='target'
        )
