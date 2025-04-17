import pytest
import os
import pandas as pd
import numpy as np
from sygnals_nn.run import run_inference
from sygnals_nn.create import create_network
from sygnals_nn.train import train_model # Needed for dummy training
from sygnals_nn.preprocess import preprocess_data # Needed for creating preprocessor
from sygnals_nn.convert import convert_to_onnx # Needed for creating ONNX model
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Fixture to create necessary files for running inference
@pytest.fixture
def inference_setup(tmp_path):
    """Creates model, data, and optionally preprocessor/onnx model."""
    setup_files = {}

    # --- Create Basic CSV Data ---
    csv_train_path = tmp_path / "train_run.csv"
    pd.DataFrame({
        'f1': [1.0, 2.0, 3.0, 4.0], 'f2': [0.1, 0.2, 0.3, 0.4],
        'text': ['run good', 'run bad', 'run okay', 'run great'],
        'label': [1, 0, 1, 1]
    }).to_csv(csv_train_path, index=False)
    setup_files["csv_train"] = csv_train_path

    csv_infer_path = tmp_path / "infer_run.csv"
    pd.DataFrame({
        'f1': [5.0, 6.0], 'f2': [0.5, 0.6],
        'text': ['run new good', 'run new bad']
    }).to_csv(csv_infer_path, index=False)
    setup_files["csv_infer"] = csv_infer_path

    # --- Create Basic JSON Data ---
    json_infer_path = tmp_path / "infer_run.json"
    json_data = [
        {'id': 10, 'numeric_features': [7.0, 0.7], 'text_feature': 'run json good'},
        {'id': 11, 'numeric_features': [8.0, 0.8], 'text_feature': 'run json bad'}
    ]
    with open(json_infer_path, 'w') as f:
        json.dump(json_data, f)
    setup_files["json_infer"] = json_infer_path

    # --- Create Basic Keras Model (2 inputs, 1 output) ---
    keras_model_path = tmp_path / "run_model.keras"
    create_network("2,4,1", None, "relu,sigmoid", "binary_crossentropy", "adam", str(keras_model_path))
    # Dummy train to compile and save weights correctly
    train_model(str(keras_model_path), str(csv_train_path), 1, 2, 0.01, 'f1,f2', 'label')
    setup_files["keras_model"] = keras_model_path

    # --- Create Output Path ---
    setup_files["output"] = tmp_path / "predictions_run.csv"

    return setup_files


# --- Test Keras Inference ---
def test_run_inference_keras_csv(inference_setup):
    """Test inference with Keras model on CSV data."""
    run_inference(
        model_path=str(inference_setup["keras_model"]),
        input_data_path=str(inference_setup["csv_infer"]),
        output_path=str(inference_setup["output"]),
        input_cols_str='f1,f2', # Use names
        json_input_key='features', # Default, not used for CSV
        preprocessor_path=None
    )
    assert inference_setup["output"].exists()
    preds = pd.read_csv(inference_setup["output"], header=None)
    assert preds.shape[0] == 2 # Should match number of rows in infer_run.csv
    assert preds.shape[1] == 1 # Model has 1 output neuron

def test_run_inference_keras_json(inference_setup):
    """Test inference with Keras model on JSON data."""
    # Model expects 2 features, JSON has 'numeric_features' key
    run_inference(
        model_path=str(inference_setup["keras_model"]),
        input_data_path=str(inference_setup["json_infer"]),
        output_path=str(inference_setup["output"]),
        input_cols_str='numeric_features', # Specify the key containing the list of features
        json_input_key='numeric_features', # Redundant here, but good practice
        preprocessor_path=None
    )
    assert inference_setup["output"].exists()
    preds = pd.read_csv(inference_setup["output"], header=None)
    assert preds.shape[0] == 2 # Should match number of objects in infer_run.json
    assert preds.shape[1] == 1

# --- Test ONNX Inference ---
@pytest.mark.skipif(not pytest.importorskip("onnxruntime"), reason="onnxruntime not installed")
def test_run_inference_onnx_csv(inference_setup, tmp_path):
    """Test inference with ONNX model on CSV data."""
    # Convert Keras model to ONNX
    onnx_model_path = tmp_path / "run_model.onnx"
    convert_to_onnx(str(inference_setup["keras_model"]), str(onnx_model_path))
    assert onnx_model_path.exists()

    run_inference(
        model_path=str(onnx_model_path), # Use ONNX model
        input_data_path=str(inference_setup["csv_infer"]),
        output_path=str(inference_setup["output"]),
        input_cols_str='0,1', # Use indices
        json_input_key='features',
        preprocessor_path=None
    )
    assert inference_setup["output"].exists()
    preds = pd.read_csv(inference_setup["output"], header=None)
    assert preds.shape[0] == 2
    assert preds.shape[1] == 1

# --- Test Inference with Preprocessor ---
def test_run_inference_with_scaler(inference_setup, tmp_path):
    """Test inference using a saved StandardScaler."""
    # 1. Create and save a scaler preprocessor
    scaler_path = tmp_path / "scaler.joblib"
    preprocess_data(
        input_path=str(inference_setup["csv_train"]),
        output_data_path=str(tmp_path / "scaled_train_data.csv"), # Not used here
        output_preprocessor_path=str(scaler_path),
        method='scale',
        feature_cols_str='f1,f2' # Scale the same columns the model expects
    )
    assert scaler_path.exists()

    # 2. Run inference using the scaler
    run_inference(
        model_path=str(inference_setup["keras_model"]),
        input_data_path=str(inference_setup["csv_infer"]),
        output_path=str(inference_setup["output"]),
        input_cols_str='f1,f2', # Specify columns to be scaled
        json_input_key='features',
        preprocessor_path=str(scaler_path) # Provide the scaler
    )
    assert inference_setup["output"].exists()
    preds = pd.read_csv(inference_setup["output"], header=None)
    assert preds.shape[0] == 2

def test_run_inference_with_tfidf(inference_setup, tmp_path):
    """Test inference using a saved TfidfVectorizer."""
    # 1. Create and save TF-IDF vectorizer
    vectorizer_path = tmp_path / "tfidf.joblib"
    preprocess_data(
        input_path=str(inference_setup["csv_train"]),
        output_data_path=str(tmp_path / "tfidf_train_data.csv"), # Not used here
        output_preprocessor_path=str(vectorizer_path),
        method='tfidf',
        text_col='text' # Vectorize the 'text' column
    )
    assert vectorizer_path.exists()
    vectorizer = joblib.load(vectorizer_path)
    n_features = len(vectorizer.vocabulary_)

    # 2. Create a Keras model compatible with TF-IDF output
    tfidf_model_path = tmp_path / "tfidf_model.keras"
    create_network(f"{n_features},8,1", None, "relu,sigmoid", "binary_crossentropy", "adam", str(tfidf_model_path))
    # Dummy training (Ideally train on actual TF-IDF processed data)
    # For test purposes, just ensure the model exists and loads

    # 3. Run inference using the vectorizer
    run_inference(
        model_path=str(tfidf_model_path),
        input_data_path=str(inference_setup["csv_infer"]), # Raw CSV with 'text' column
        output_path=str(inference_setup["output"]),
        input_cols_str='text', # Crucial: Specify the *original* text column name
        json_input_key='features',
        preprocessor_path=str(vectorizer_path) # Provide the vectorizer
    )
    assert inference_setup["output"].exists()
    preds = pd.read_csv(inference_setup["output"], header=None)
    assert preds.shape[0] == 2

# --- Test Error Handling ---
def test_run_inference_model_not_found(inference_setup):
    """Test error when model file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        run_inference(
            model_path="non_existent_model.keras",
            input_data_path=str(inference_setup["csv_infer"]),
            output_path=str(inference_setup["output"]),
            input_cols_str='f1,f2'
        )

def test_run_inference_data_not_found(inference_setup):
    """Test error when input data file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        run_inference(
            model_path=str(inference_setup["keras_model"]),
            input_data_path="non_existent_data.csv",
            output_path=str(inference_setup["output"]),
            input_cols_str='f1,f2'
        )

def test_run_inference_preprocessor_not_found(inference_setup):
    """Test error when preprocessor file doesn't exist but is specified."""
    with pytest.raises(FileNotFoundError):
        run_inference(
            model_path=str(inference_setup["keras_model"]),
            input_data_path=str(inference_setup["csv_infer"]),
            output_path=str(inference_setup["output"]),
            input_cols_str='f1,f2',
            preprocessor_path="non_existent_preprocessor.joblib"
        )

def test_run_inference_missing_input_cols(inference_setup):
    """Test error when input columns are not specified."""
    # This should now raise an error in load_data if preprocessor doesn't handle it
    with pytest.raises(ValueError, match="Input columns must be specified"):
        run_inference(
            model_path=str(inference_setup["keras_model"]),
            input_data_path=str(inference_setup["csv_infer"]),
            output_path=str(inference_setup["output"]),
            input_cols_str=None # Missing input_cols
        )
