import tensorflow as tf
import numpy as np
import pandas as pd
import os
import logging
import joblib # For loading preprocessor objects

# Try importing ONNX runtime, warn if not available but allow Keras inference
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    logging.warning("onnxruntime not found. ONNX model inference will not be available.")

from sygnals_nn.utils import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_inference(
    model_path: str,
    input_data_path: str,
    output_path: str | None,
    input_cols_str: str,
    json_input_key: str = 'features',
    preprocessor_path: str | None = None
    ):
    """
    Runs inference using a trained Keras or ONNX model.

    Args:
        model_path: Path to the trained Keras (.keras) or ONNX (.onnx) model file.
        input_data_path: Path to the input data file (CSV or JSON) for inference.
        output_path: Optional path to save predictions (CSV format). Prints to console if None.
        input_cols_str: Comma-separated input column indices or names.
        json_input_key: Key for input features in JSON data.
        preprocessor_path: Optional path to a saved preprocessor object (e.g., scaler, vectorizer)
                           to apply to the input data before inference.

    Raises:
        FileNotFoundError: If model, data, or preprocessor file not found.
        ValueError: If model type is unsupported or ONNX runtime is needed but unavailable.
        Exception: For errors during loading, preprocessing, or inference.
    """
    logging.info(f"Running inference with model: {model_path}")
    logging.info(f"Input data: {input_data_path}")
    if preprocessor_path:
        logging.info(f"Using preprocessor: {preprocessor_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(input_data_path):
        raise FileNotFoundError(f"Input data file not found: {input_data_path}")
    if preprocessor_path and not os.path.exists(preprocessor_path):
         raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")


    # --- Step 1: Load Preprocessor (if specified) ---
    preprocessor = None
    text_col_for_preprocess = None # Will be set if preprocessor is text-based
    if preprocessor_path:
        try:
            logging.info(f"Loading preprocessor from {preprocessor_path}...")
            preprocessor = joblib.load(preprocessor_path)
            logging.info(f"Preprocessor loaded successfully: {type(preprocessor)}")
            # Identify if it's a text vectorizer to pass the text column name to load_data
            # This check might need refinement based on actual preprocessor types used
            if hasattr(preprocessor, 'vocabulary_') or 'TfidfVectorizer' in str(type(preprocessor)) or 'CountVectorizer' in str(type(preprocessor)):
                 # We need to know which *original* column the vectorizer was trained on.
                 # This info isn't typically stored in the vectorizer itself.
                 # Assumption: If a text vectorizer is used, input_cols_str likely refers
                 # to the *original text column* name or index.
                 logging.warning("Assuming --input-cols refers to the original text column for the loaded text vectorizer.")
                 text_col_for_preprocess = input_cols_str # Pass the column name/index string
                 input_cols_str = None # Prevent load_data from trying to select it as a feature column
            elif not input_cols_str:
                 raise ValueError("If using a non-text preprocessor (like a scaler), --input-cols must still be provided to select the columns to process.")

        except Exception as e:
            logging.error(f"Error loading preprocessor from {preprocessor_path}: {e}")
            raise


    # --- Step 2: Load and Preprocess Input Data ---
    try:
        logging.info("Loading input data for inference...")
        # Pass the loaded preprocessor to load_data
        X_input, _ = load_data(
            file_path=input_data_path,
            input_cols_str=input_cols_str, # Will be None if text_col_for_preprocess is set
            label_cols_str=None, # No labels needed for inference
            json_input_key=json_input_key,
            is_inference=True,
            preprocessor=preprocessor, # Pass the loaded object
            text_col_for_preprocess=text_col_for_preprocess # Pass the text column name if applicable
        )
        logging.info(f"Input data loaded and preprocessed. X_input shape: {X_input.shape}")
        if X_input.shape[0] == 0:
            raise ValueError("Loaded input data is empty after processing.")

    except Exception as e:
        logging.error(f"Error loading or preprocessing input data from {input_data_path}: {e}")
        raise


    # --- Step 3: Load Model and Perform Inference ---
    predictions = None
    model_type = "keras" if model_path.lower().endswith(".keras") else "onnx" if model_path.lower().endswith(".onnx") else "unknown"

    if model_type == "keras":
        try:
            logging.info("Loading Keras model for inference...")
            model = tf.keras.models.load_model(model_path)
            logging.info("Keras model loaded.")
            # Ensure input data type matches model expectation (usually float32)
            if X_input.dtype != tf.float32:
                 logging.warning(f"Input data dtype is {X_input.dtype}, converting to float32 for Keras model.")
                 X_input = X_input.astype(np.float32)

            logging.info("Performing Keras inference...")
            predictions = model.predict(X_input)
            logging.info("Keras inference completed.")

        except Exception as e:
            logging.error(f"Error during Keras model loading or inference: {e}")
            raise

    elif model_type == "onnx":
        if not ORT_AVAILABLE:
            raise ValueError("ONNX model specified, but onnxruntime library is not installed.")
        try:
            logging.info("Loading ONNX model for inference...")
            ort_session = ort.InferenceSession(model_path)
            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name
            logging.info(f"ONNX model loaded. Input: '{input_name}', Output: '{output_name}'")

             # Ensure input data type matches ONNX model expectation (often float32)
            input_type = ort_session.get_inputs()[0].type
            expected_dtype_str = input_type.split('(')[-1].split(')')[0] # e.g., 'float' from 'tensor(float)'
            expected_dtype = np.float32 # Default assumption
            if 'float' in expected_dtype_str: expected_dtype = np.float32
            elif 'double' in expected_dtype_str: expected_dtype = np.float64
            elif 'int64' in expected_dtype_str: expected_dtype = np.int64
            elif 'int32' in expected_dtype_str: expected_dtype = np.int32
            # Add more types as needed

            if X_input.dtype != expected_dtype:
                 logging.warning(f"Input data dtype is {X_input.dtype}, converting to {expected_dtype} for ONNX model.")
                 X_input = X_input.astype(expected_dtype)


            logging.info("Performing ONNX inference...")
            predictions = ort_session.run([output_name], {input_name: X_input})[0] # [0] because run returns a list of outputs
            logging.info("ONNX inference completed.")

        except Exception as e:
            logging.error(f"Error during ONNX model loading or inference: {e}")
            raise
    else:
        raise ValueError(f"Unsupported model file type: {model_path}. Use .keras or .onnx.")


    # --- Step 4: Save or Print Predictions ---
    if predictions is None:
         logging.error("Inference did not produce predictions.")
         return # Or raise error

    logging.info(f"Predictions shape: {predictions.shape}")

    if output_path:
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Save as CSV without header or index
            pd.DataFrame(predictions).to_csv(output_path, index=False, header=False)
            logging.info(f"Predictions successfully saved to {output_path}")
        except Exception as e:
            logging.error(f"Error saving predictions to {output_path}: {e}")
            # Optionally print predictions if saving fails
            print("--- Predictions (Saving Failed) ---")
            print(predictions)
            print("------------------------------------")
    else:
        # Print predictions to the console if no output file specified
        print("--- Predictions ---")
        # Use pandas for nice formatting, limit rows shown for large outputs
        print(pd.DataFrame(predictions).to_string(max_rows=20))
        if predictions.shape[0] > 20:
             print(f"... (truncated, {predictions.shape[0]} total predictions)")
        print("-------------------")
