import tensorflow as tf
import tf2onnx # Or import keras_onnx if using that library
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_to_onnx(
    keras_model_path: str,
    output_onnx_path: str,
    input_signature_str: str | None = None,
    input_signature_list: list | None = None, # Allow passing parsed signature directly
    opset: int = 13 # Default ONNX opset version, adjust if needed
    ):
    """
    Converts a saved Keras model (.keras) to ONNX format (.onnx).

    Args:
        keras_model_path: Path to the input Keras model file.
        output_onnx_path: Path to save the output ONNX model file.
        input_signature_str: Optional string representation of the input signature.
                             Example: "[tf.TensorSpec(shape=(None, 784), dtype=tf.float32)]"
                             This will be evaluated. Use with caution.
        input_signature_list: Optional list of tf.TensorSpec objects. Takes precedence
                              over input_signature_str if provided. Inferred if both are None.
        opset: The ONNX opset version to target.

    Raises:
        FileNotFoundError: If the Keras model file doesn't exist.
        ImportError: If tf2onnx is not installed.
        Exception: For errors during model loading or conversion.
    """
    logging.info(f"Starting ONNX conversion for: {keras_model_path}")
    logging.info(f"Target ONNX output: {output_onnx_path}")

    if not os.path.exists(keras_model_path):
        raise FileNotFoundError(f"Keras model file not found: {keras_model_path}")

    # --- Load Keras Model ---
    try:
        logging.info("Loading Keras model...")
        model = tf.keras.models.load_model(keras_model_path)
        logging.info("Keras model loaded successfully.")
        model.summary(print_fn=logging.info)
    except Exception as e:
        logging.error(f"Error loading Keras model from {keras_model_path}: {e}")
        raise

    # --- Determine Input Signature ---
    input_signature = None
    if input_signature_list:
        input_signature = input_signature_list
        logging.info(f"Using provided input signature list: {input_signature}")
    elif input_signature_str:
        try:
            # WARNING: eval can be dangerous if the string is user-controlled.
            # Ensure the input string is validated or trusted.
            logging.warning(f"Evaluating input signature string: {input_signature_str}")
            # Make tf available in the eval context
            eval_context = {'tf': tf}
            input_signature = eval(input_signature_str, eval_context)
            if not isinstance(input_signature, list):
                 input_signature = [input_signature] # Ensure it's a list
            logging.info(f"Using evaluated input signature: {input_signature}")
        except Exception as e:
            logging.error(f"Error evaluating input_signature_str: {e}. Please provide a valid Python list of tf.TensorSpec objects.")
            raise ValueError("Invalid input_signature_str format.") from e
    else:
        # Try to infer signature from the loaded model
        try:
            logging.info("Attempting to infer input signature from model...")
            # tf.keras.layers.InputLayer is often the first layer implicitly
            input_spec = model.inputs[0].spec
            input_signature = [input_spec]
            logging.info(f"Inferred input signature: {input_signature}")
        except Exception as e:
            logging.warning(f"Could not automatically infer input signature: {e}. "
                            "ONNX conversion might fail or produce an invalid model. "
                            "Consider providing --input-signature explicitly.")
            # Conversion might still work for some simple models without it, but it's risky.
            # input_signature = None # Proceed without signature if inference fails

    if not input_signature:
         logging.warning("Proceeding with ONNX conversion without a defined input signature. This may fail.")


    # --- Perform ONNX Conversion ---
    try:
        logging.info(f"Converting model to ONNX (opset {opset})...")
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)

        # Use tf2onnx.convert.from_keras
        # Pass the loaded model and the determined input signature
        model_proto, external_tensor_storage = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=opset,
            output_path=output_onnx_path # tf2onnx can save directly
        )
        logging.info(f"Model successfully converted and saved to ONNX: {output_onnx_path}")

    except ImportError:
        logging.error("`tf2onnx` library is required for ONNX conversion. Please install it (`pip install tf2onnx`).")
        raise
    except Exception as e:
        logging.error(f"Error during ONNX conversion: {e}")
        logging.error("Ensure the input signature (if provided or inferred) matches the model's input layer.")
        logging.error("Check compatibility between TensorFlow, Keras, and tf2onnx versions.")
        raise
