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

# Try importing TensorFlow Probability for potential sampling/distribution use
try:
    import tensorflow_probability as tfp
    TFP_AVAILABLE = True
except ImportError:
    TFP_AVAILABLE = False
    logging.debug("TensorFlow Probability (TFP) not found. Direct sampling from distributions in run.py might be limited.")


from sygnals_nn.utils import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_inference(
    model_path: str,
    input_data_path: str,
    output_path: str | None,
    input_cols_str: str,
    json_input_key: str = 'features',
    preprocessor_path: str | None = None,
    prediction_mode: str = 'params', # New: 'params', 'samples', 'mean_stddev'
    mc_dropout: bool = False,        # New: Flag for MC Dropout
    num_samples: int = 30            # New: Number of samples for MC or probabilistic sampling
    ):
    """
    Runs inference using a trained Keras or ONNX model.
    Handles deterministic and probabilistic (Gaussian regression for Keras) model outputs.

    Args:
        model_path (str): Path to the trained Keras (.keras) or ONNX (.onnx) model file.
        input_data_path (str): Path to the input data file (CSV or JSON) for inference.
        output_path (str | None): Optional path to save predictions (CSV format).
                                  Prints to console if None.
        input_cols_str (str): Comma-separated input column indices or names.
        json_input_key (str): Key for input features in JSON data. Defaults to 'features'.
        preprocessor_path (str | None): Optional path to a saved preprocessor object
                                       (e.g., scaler, vectorizer) to apply to the
                                       input data before inference. Defaults to None.
        prediction_mode (str): Mode for probabilistic predictions.
                               - 'params': Output raw distribution parameters (e.g., mean, log_variance).
                               - 'samples': Output multiple samples drawn from the predicted distribution
                                            (MC Dropout or model's distribution). (MC Dropout part Phase 2)
                               - 'mean_stddev': Output mean and standard deviation.
                               Defaults to 'params'.
        mc_dropout (bool): If True, enables Monte Carlo Dropout for uncertainty estimation
                           if the model contains dropout layers and is a Keras model.
                           (Full implementation in Phase 2). Defaults to False.
        num_samples (int): Number of samples to generate for MC Dropout or when
                           `prediction_mode` is 'samples'. Defaults to 30.

    Raises:
        FileNotFoundError: If model, data, or preprocessor file not found.
        ValueError: If model type is unsupported, ONNX runtime is needed but unavailable,
                    or if `prediction_mode` is 'samples' and TFP is unavailable for Keras models.
        Exception: For errors during loading, preprocessing, or inference.
    """
    logging.info(f"Running inference with model: {model_path}")
    logging.info(f"Input data: {input_data_path}")
    logging.info(f"Prediction mode: {prediction_mode}, MC Dropout: {mc_dropout}, Num samples: {num_samples}")
    if preprocessor_path:
        logging.info(f"Using preprocessor: {preprocessor_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(input_data_path):
        raise FileNotFoundError(f"Input data file not found: {input_data_path}")
    if preprocessor_path and not os.path.exists(preprocessor_path):
         raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")

    preprocessor = None
    text_col_for_preprocess = None
    if preprocessor_path:
        try:
            logging.info(f"Loading preprocessor from {preprocessor_path}...")
            preprocessor = joblib.load(preprocessor_path)
            logging.info(f"Preprocessor loaded successfully: {type(preprocessor)}")
            if hasattr(preprocessor, 'vocabulary_') or 'TfidfVectorizer' in str(type(preprocessor)) or 'CountVectorizer' in str(type(preprocessor)):
                 logging.info("Detected text vectorizer. --input-cols will specify the text column in raw data.")
                 text_col_for_preprocess = input_cols_str
                 input_cols_str = None # load_data will use text_col_for_preprocess
            elif not input_cols_str:
                 raise ValueError("If using a non-text preprocessor (e.g., scaler), --input-cols must still be provided to select the columns to process.")
        except Exception as e:
            logging.error(f"Error loading preprocessor from {preprocessor_path}: {e}")
            raise

    try:
        logging.info("Loading input data for inference...")
        X_input, _ = load_data(
            file_path=input_data_path,
            input_cols_str=input_cols_str,
            label_cols_str=None, # No labels needed for inference
            json_input_key=json_input_key,
            is_inference=True,
            preprocessor=preprocessor,
            text_col_for_preprocess=text_col_for_preprocess
        )
        logging.info(f"Input data loaded and preprocessed. X_input shape: {X_input.shape}")
        if X_input.shape[0] == 0:
            raise ValueError("Loaded input data is empty after processing.")
    except Exception as e:
        logging.error(f"Error loading or preprocessing input data from {input_data_path}: {e}")
        raise

    predictions_df = None
    model_type = "keras" if model_path.lower().endswith(".keras") else "onnx" if model_path.lower().endswith(".onnx") else "unknown"

    if model_type == "keras":
        try:
            logging.info("Loading Keras model for inference...")
            # Load model with custom objects if necessary (e.g., for custom NLL loss if saved with it)
            # For now, assume standard loading is fine as loss is applied at train time.
            model = tf.keras.models.load_model(model_path, compile=False) # Re-compile not needed for inference
            logging.info("Keras model loaded.")
            model.summary(print_fn=logging.debug) # Log summary at debug level

            if X_input.dtype != tf.float32:
                 logging.warning(f"Input data dtype is {X_input.dtype}, converting to float32 for Keras model.")
                 X_input = X_input.astype(np.float32)

            if mc_dropout:
                # Full MC Dropout implementation is Phase 2.
                # This requires calling the model multiple times with dropout layers active.
                # model.predict() does not have a `training` argument.
                # One would typically call `model(X_input, training=True)` in a loop.
                logging.warning("MC Dropout (--mc-dropout) for Keras models is specified.")
                logging.warning("True MC Dropout requires multiple forward passes with dropout layers active (training=True).")
                logging.warning("Current implementation will proceed with standard prediction. Full MC Dropout is targeted for Phase 2.")
                # For now, proceed with a single prediction pass.
                raw_predictions = model.predict(X_input)
            else:
                logging.info("Performing Keras inference...")
                raw_predictions = model.predict(X_input)
                logging.info("Keras inference completed.")

            # --- Process predictions for Keras models (especially probabilistic) ---
            num_output_units = model.output_shape[-1]
            # Heuristic: if output units are even, and mode is params/mean_stddev, assume Gaussian output
            # This should ideally be based on metadata saved with the model.
            is_likely_gaussian_output = (num_output_units > 0 and num_output_units % 2 == 0 and
                                         prediction_mode in ['params', 'mean_stddev'])

            if is_likely_gaussian_output and not mc_dropout : # Don't apply if MC samples are the raw_predictions
                num_target_dimensions = num_output_units // 2
                logging.info(f"Keras model output suggests {num_target_dimensions} target dimension(s) for Gaussian output (mean & log_variance).")

                means = raw_predictions[:, :num_target_dimensions]
                log_variances = raw_predictions[:, num_target_dimensions:]

                if prediction_mode == 'params':
                    pred_data = {}
                    for i in range(num_target_dimensions):
                        pred_data[f'mean_{i}'] = means[:, i]
                        pred_data[f'log_variance_{i}'] = log_variances[:, i]
                    predictions_df = pd.DataFrame(pred_data)
                    logging.info("Outputting mean and log_variance parameters.")
                elif prediction_mode == 'mean_stddev':
                    std_devs = np.exp(0.5 * log_variances)
                    pred_data = {}
                    for i in range(num_target_dimensions):
                        pred_data[f'mean_{i}'] = means[:, i]
                        pred_data[f'stddev_{i}'] = std_devs[:, i]
                    predictions_df = pd.DataFrame(pred_data)
                    logging.info("Outputting mean and standard deviation.")
                elif prediction_mode == 'samples':
                    if not TFP_AVAILABLE:
                        raise ValueError("TensorFlow Probability (TFP) is required for 'samples' mode with Gaussian Keras models.")
                    logging.info(f"Generating {num_samples} samples per input from predicted Gaussian distributions...")
                    # This part would involve tfp.distributions.Normal(loc=means, scale=np.exp(0.5 * log_variances)).sample(num_samples)
                    # and then reshaping/aggregating. For simplicity in this step, we'll log a TO-DO.
                    logging.warning("'samples' mode for direct Gaussian output is not fully implemented yet. Outputting raw parameters instead.")
                    predictions_df = pd.DataFrame(raw_predictions) # Fallback to raw
                else: # Should not happen due to CLI choices
                    predictions_df = pd.DataFrame(raw_predictions)

            else: # Deterministic model or MC Dropout (raw samples) or other cases
                predictions_df = pd.DataFrame(raw_predictions)
                if mc_dropout:
                     logging.info("MC Dropout produced raw prediction samples (further aggregation if needed is manual for now).")


        except Exception as e:
            logging.error(f"Error during Keras model loading or inference: {e}", exc_info=True)
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

            input_type_info = ort_session.get_inputs()[0].type
            expected_dtype_str = input_type_info.split('(')[-1].split(')')[0] if '(' in input_type_info else 'float'
            expected_dtype = np.float32
            if 'double' in expected_dtype_str: expected_dtype = np.float64
            elif 'int64' in expected_dtype_str: expected_dtype = np.int64
            elif 'int32' in expected_dtype_str: expected_dtype = np.int32
            # Add more types as needed, default to float32 for 'float' or unknown

            if X_input.dtype != expected_dtype:
                 logging.warning(f"Input data dtype is {X_input.dtype}, converting to {expected_dtype} for ONNX model.")
                 X_input = X_input.astype(expected_dtype)

            if mc_dropout:
                logging.warning("MC Dropout (--mc-dropout) is specified for an ONNX model. ONNX runtimes typically do not support a 'training mode' for dropout layers. Standard inference will be performed.")

            logging.info("Performing ONNX inference...")
            # ONNX run returns a list of outputs; typically one for most NNs.
            raw_predictions = ort_session.run([output_name], {input_name: X_input})[0]
            logging.info("ONNX inference completed.")

            # For ONNX, probabilistic interpretation (mean/stddev/params) is not yet implemented here.
            # We output the raw predictions.
            if prediction_mode != 'params': # Or whatever the raw output is considered
                 logging.warning(f"Prediction mode '{prediction_mode}' for ONNX probabilistic models is not fully supported yet. Outputting raw model predictions.")
            predictions_df = pd.DataFrame(raw_predictions)


        except Exception as e:
            logging.error(f"Error during ONNX model loading or inference: {e}", exc_info=True)
            raise
    else:
        raise ValueError(f"Unsupported model file type: {model_path}. Use .keras or .onnx.")

    if predictions_df is None:
         logging.error("Inference did not produce predictions DataFrame.")
         return

    logging.info(f"Predictions DataFrame shape: {predictions_df.shape}")
    logging.debug(f"Predictions head:\n{predictions_df.head().to_string()}")


    if output_path:
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir: os.makedirs(output_dir, exist_ok=True)
            # Save with header if predictions_df has meaningful column names (from probabilistic processing)
            # Otherwise, save without header for raw/deterministic outputs.
            save_header = not all(isinstance(col, int) for col in predictions_df.columns)
            predictions_df.to_csv(output_path, index=False, header=save_header)
            logging.info(f"Predictions successfully saved to {output_path} (header: {save_header})")
        except Exception as e:
            logging.error(f"Error saving predictions to {output_path}: {e}")
            print("\n--- Predictions (Saving Failed) ---")
            print(predictions_df.to_string(max_rows=20))
            if predictions_df.shape[0] > 20: print(f"... (truncated, {predictions_df.shape[0]} total predictions)")
            print("------------------------------------\n")
    else:
        print("\n--- Predictions ---")
        # Use pandas for nice formatting, limit rows shown for large outputs
        print(predictions_df.to_string(max_rows=20))
        if predictions_df.shape[0] > 20:
             print(f"... (truncated, {predictions_df.shape[0]} total predictions)")
        print("-------------------\n")
