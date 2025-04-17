import pytest
import os
import tensorflow as tf
import pandas as pd
from sygnals_nn.create import create_network
from sygnals_nn.convert import convert_to_onnx

# Skip all tests in this module if tf2onnx is not installed
pytest.importorskip("tf2onnx")
# Also skip if onnxruntime is not installed, as it's useful for verification
ort = pytest.importorskip("onnxruntime")


# Fixture to create a Keras model for conversion
@pytest.fixture
def keras_model(tmp_path):
    """Creates and saves a simple Keras model."""
    model_path = tmp_path / "convert_model.keras"
    create_network(
        layers_str="5,10,1", # Input=5, Hidden=10, Output=1
        layer_types_str=None, # Dense
        activations_str="relu,sigmoid",
        loss="binary_crossentropy",
        optimizer="adam",
        output_path=str(model_path)
    )
    assert model_path.exists()
    return model_path

# Fixture for ONNX output path
@pytest.fixture
def onnx_output_path(tmp_path):
    """Provides a path for the output ONNX model."""
    return tmp_path / "converted_model.onnx"


# --- Test Successful Conversion ---
def test_convert_to_onnx_success(keras_model, onnx_output_path):
    """Test basic successful conversion from Keras to ONNX."""
    convert_to_onnx(
        keras_model_path=str(keras_model),
        output_onnx_path=str(onnx_output_path)
        # Rely on inferred signature
    )
    assert onnx_output_path.exists()

    # Basic verification: Try loading the ONNX model
    try:
        ort_session = ort.InferenceSession(str(onnx_output_path))
        assert ort_session is not None
        assert len(ort_session.get_inputs()) > 0 # Check if it has inputs
    except Exception as e:
        pytest.fail(f"Failed to load the converted ONNX model: {e}")

def test_convert_to_onnx_with_explicit_signature(keras_model, onnx_output_path):
    """Test conversion providing an explicit input signature string."""
    # Signature must match the model created in the fixture (input=5)
    input_sig_str = "[tf.TensorSpec(shape=(None, 5), dtype=tf.float32)]"
    convert_to_onnx(
        keras_model_path=str(keras_model),
        output_onnx_path=str(onnx_output_path),
        input_signature_str=input_sig_str,
        opset=15 # Test with a different opset
    )
    assert onnx_output_path.exists()
    # Verify load
    ort_session = ort.InferenceSession(str(onnx_output_path))
    assert ort_session is not None
    # Check input shape from ONNX model if possible (more advanced check)
    input_meta = ort_session.get_inputs()[0]
    assert input_meta.shape == [None, 5] # Check if shape matches signature


# --- Test Error Handling ---
def test_convert_keras_model_not_found(onnx_output_path):
    """Test error when the input Keras model file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        convert_to_onnx(
            keras_model_path="non_existent.keras",
            output_onnx_path=str(onnx_output_path)
        )

def test_convert_invalid_signature_string(keras_model, onnx_output_path):
    """Test error when the provided signature string is invalid."""
    with pytest.raises(ValueError, match="Invalid input_signature_str format"):
        convert_to_onnx(
            keras_model_path=str(keras_model),
            output_onnx_path=str(onnx_output_path),
            input_signature_str="this is not a valid signature"
        )

# Note: Testing signature *mismatch* errors during conversion can be tricky
# as tf2onnx might sometimes succeed even with a slightly off signature,
# potentially leading to runtime errors later. The explicit signature test
# above provides some confidence.
