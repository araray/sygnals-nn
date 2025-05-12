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
    prediction_mode: str = 'params',
    mc_dropout: bool = False,
    num_samples: int = 30
    ):
    """
    Runs inference using a trained Keras or ONNX model.
    Handles deterministic and probabilistic (Gaussian regression for Keras) model outputs,
    and supports Monte Carlo (MC) Dropout for Keras models.

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
                               - 'params': Output raw distribution parameters (e.g., Keras Gaussian: mean, log_variance).
                               - 'samples': Output multiple samples drawn from the predicted distribution
                                            (e.g., from MC Dropout).
                               - 'mean_stddev': Output mean and standard deviation (from MC Dropout samples
                                                or Keras Gaussian parameters).
                               Defaults to 'params'.
        mc_dropout (bool): If True, enables Monte Carlo Dropout for uncertainty estimation
                           if the model is a Keras model and contains dropout layers.
                           Defaults to False.
        num_samples (int): Number of samples to generate for MC Dropout or when
                           `prediction_mode` is 'samples' (for future direct model sampling).
                           Defaults to 30.

    Raises:
        FileNotFoundError: If model, data, or preprocessor file not found.
        ValueError: If model type is unsupported, ONNX runtime is needed but unavailable,
                    or if `prediction_mode` is 'samples' and TFP is unavailable for Keras Gaussian models.
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
                 input_cols_str = None
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
            label_cols_str=None,
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
            model = tf.keras.models.load_model(model_path, compile=False)
            logging.info("Keras model loaded.")
            model.summary(print_fn=logging.debug)

            if X_input.dtype != tf.float32:
                 logging.warning(f"Input data dtype is {X_input.dtype}, converting to float32 for Keras model.")
                 X_input = X_input.astype(np.float32)

            raw_predictions_list = [] # To store multiple prediction passes for MC Dropout

            if mc_dropout:
                logging.info(f"Performing MC Dropout with {num_samples} samples...")
                # Check if the model has dropout layers. This is a basic check;
                # a more robust check would inspect layer types.
                has_dropout = any('dropout' in layer.__class__.__name__.lower() for layer in model.layers)
                if not has_dropout:
                    logging.warning("MC Dropout enabled, but no Dropout layers found in the Keras model. Performing standard prediction.")
                    # Fallback to standard prediction if no dropout layers
                    raw_predictions = model.predict(X_input)
                    raw_predictions_list.append(raw_predictions)
                else:
                    for i in range(num_samples):
                        logging.debug(f"MC Dropout sample {i+1}/{num_samples}")
                        # Call the model with training=True to activate dropout layers
                        # This assumes the model's call method handles the training flag appropriately.
                        y_pred_sample = model(X_input, training=True)
                        raw_predictions_list.append(y_pred_sample.numpy()) # Convert EagerTensor to numpy

                    # Stack predictions: from list of (batch, features) to (samples, batch, features)
                    # Then transpose to (batch, samples, features) for easier aggregation per input instance
                    if raw_predictions_list:
                        raw_predictions_stacked = np.stack(raw_predictions_list, axis=0)
                        raw_predictions_transposed = np.transpose(raw_predictions_stacked, (1, 0, 2))
                        logging.info(f"MC Dropout collected predictions. Shape: (batch_size, num_samples, num_outputs_per_sample) = {raw_predictions_transposed.shape}")
                    else: # Should not happen if loop runs
                        logging.error("MC Dropout loop did not produce any predictions.")
                        raw_predictions_transposed = np.array([]).reshape(X_input.shape[0], 0, model.output_shape[-1])


            else: # Standard single-pass inference
                logging.info("Performing standard Keras inference (single pass)...")
                raw_predictions = model.predict(X_input)
                raw_predictions_list.append(raw_predictions) # Store as a list for consistent processing below
                # For standard prediction, shape is (batch, features), so transpose to (batch, 1, features)
                raw_predictions_transposed = raw_predictions[:, np.newaxis, :] if raw_predictions.ndim == 2 else raw_predictions # Handle multi-dim output carefully
                if raw_predictions.ndim == 1: # for single output regression (batch,)
                    raw_predictions_transposed = raw_predictions[:, np.newaxis, np.newaxis]


            # --- Process predictions for Keras models ---
            num_output_units_per_sample = model.output_shape[-1]

            if mc_dropout and has_dropout: # Process MC Dropout samples
                if prediction_mode == 'samples':
                    # Reshape for output: (batch_size * num_samples, num_outputs_per_sample)
                    # Or keep as (batch_size, num_samples, num_outputs_per_sample) and create multi-index?
                    # For now, let's create columns for each sample and each output unit
                    # E.g., sample_0_output_0, sample_0_output_1, ..., sample_1_output_0, ...
                    # This can lead to many columns if num_samples or num_outputs_per_sample is large.
                    # A more common way is to output mean/stddev or provide samples in a long format.
                    # Let's try (batch_size, num_samples * num_outputs_per_sample)
                    predictions_reshaped = raw_predictions_transposed.reshape(X_input.shape[0], -1)
                    pred_cols = [f'sample_{s}_output_{o}' for s in range(num_samples) for o in range(num_output_units_per_sample)]
                    predictions_df = pd.DataFrame(predictions_reshaped, columns=pred_cols[:predictions_reshaped.shape[1]])
                    logging.info(f"Outputting {num_samples} MC Dropout samples per input instance.")

                elif prediction_mode == 'mean_stddev':
                    means = np.mean(raw_predictions_transposed, axis=1)
                    stddevs = np.std(raw_predictions_transposed, axis=1)
                    pred_data = {}
                    for i in range(num_output_units_per_sample):
                        # Naming depends on whether it's regression or classification probabilities
                        if num_output_units_per_sample > 1: # Likely classification probs or multi-target regression
                            pred_data[f'mean_output_{i}'] = means[:, i]
                            pred_data[f'stddev_output_{i}'] = stddevs[:, i]
                        else: # Single output regression
                            pred_data['pred_mean'] = means[:, i].flatten()
                            pred_data['pred_stddev'] = stddevs[:, i].flatten()
                    predictions_df = pd.DataFrame(pred_data)
                    logging.info("Outputting mean and standard deviation from MC Dropout samples.")
                else: # 'params' mode for MC Dropout doesn't make sense, fallback to mean/stddev
                    logging.warning(f"'params' mode is not directly applicable for MC Dropout. Outputting mean/stddev instead.")
                    means = np.mean(raw_predictions_transposed, axis=1)
                    stddevs = np.std(raw_predictions_transposed, axis=1)
                    # (Same logic as above for mean_stddev)
                    pred_data = {}
                    for i in range(num_output_units_per_sample):
                        if num_output_units_per_sample > 1:
                            pred_data[f'mean_output_{i}'] = means[:, i]
                            pred_data[f'stddev_output_{i}'] = stddevs[:, i]
                        else:
                            pred_data['pred_mean'] = means[:, i].flatten()
                            pred_data['pred_stddev'] = stddevs[:, i].flatten()
                    predictions_df = pd.DataFrame(pred_data)

            else: # Process standard Keras output (Gaussian probabilistic or deterministic)
                # This is the single pass prediction (raw_predictions_list[0])
                single_pass_predictions = raw_predictions_list[0]
                is_likely_gaussian_output = (num_output_units_per_sample > 0 and
                                             num_output_units_per_sample % 2 == 0 and
                                             prediction_mode in ['params', 'mean_stddev'])

                if is_likely_gaussian_output: # Keras model trained for Gaussian output
                    num_target_dimensions = num_output_units_per_sample // 2
                    logging.info(f"Keras model output suggests {num_target_dimensions} target dimension(s) for Gaussian output (mean & log_variance).")

                    means = single_pass_predictions[:, :num_target_dimensions]
                    log_variances = single_pass_predictions[:, num_target_dimensions:]

                    if prediction_mode == 'params':
                        pred_data = {}
                        for i in range(num_target_dimensions):
                            pred_data[f'mean_dim{i}'] = means[:, i]
                            pred_data[f'log_variance_dim{i}'] = log_variances[:, i]
                        predictions_df = pd.DataFrame(pred_data)
                        logging.info("Outputting Gaussian mean and log_variance parameters.")
                    elif prediction_mode == 'mean_stddev':
                        std_devs = np.exp(0.5 * log_variances)
                        pred_data = {}
                        for i in range(num_target_dimensions):
                            pred_data[f'mean_dim{i}'] = means[:, i]
                            pred_data[f'stddev_dim{i}'] = std_devs[:, i]
                        predictions_df = pd.DataFrame(pred_data)
                        logging.info("Outputting Gaussian mean and standard deviation.")
                    elif prediction_mode == 'samples':
                        if not TFP_AVAILABLE:
                            raise ValueError("TensorFlow Probability (TFP) is required for 'samples' mode with Gaussian Keras models.")
                        logging.info(f"Generating {num_samples} samples per input from predicted Gaussian distributions...")
                        # Sample from the distribution
                        dist = tfp.distributions.Normal(loc=means, scale=np.exp(0.5 * log_variances))
                        samples = dist.sample(num_samples).numpy() # Shape: (num_mc_samples, batch_size, num_target_dims)
                        # Transpose to (batch_size, num_mc_samples, num_target_dims) then reshape
                        samples_transposed = np.transpose(samples, (1, 0, 2))
                        samples_reshaped = samples_transposed.reshape(X_input.shape[0], -1)

                        pred_cols = [f'sample_{s}_dim_{d}' for s in range(num_samples) for d in range(num_target_dimensions)]
                        predictions_df = pd.DataFrame(samples_reshaped, columns=pred_cols[:samples_reshaped.shape[1]])
                        logging.info(f"Outputting {num_samples} samples drawn from the predicted Gaussian distributions.")
                    else: # Should not happen
                        predictions_df = pd.DataFrame(single_pass_predictions)
                else: # Deterministic Keras model output
                    predictions_df = pd.DataFrame(single_pass_predictions)
                    if predictions_df.shape[1] > 1: # Multiple outputs from deterministic model
                        predictions_df.columns = [f'output_{i}' for i in range(predictions_df.shape[1])]
                    else: # Single output
                        predictions_df.columns = ['prediction']


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

            if X_input.dtype != expected_dtype:
                 logging.warning(f"Input data dtype is {X_input.dtype}, converting to {expected_dtype} for ONNX model.")
                 X_input = X_input.astype(expected_dtype)

            if mc_dropout:
                logging.warning("MC Dropout (--mc-dropout) is specified for an ONNX model. Standard ONNX runtimes do not support a 'training mode' for dropout layers. Performing standard inference.")

            logging.info("Performing ONNX inference...")
            raw_predictions = ort_session.run([output_name], {input_name: X_input})[0]
            logging.info("ONNX inference completed.")

            if prediction_mode != 'params' and prediction_mode != 'samples': # 'samples' for ONNX is just raw output
                 logging.warning(f"Prediction mode '{prediction_mode}' for ONNX probabilistic models is not fully supported for parameter interpretation. Outputting raw model predictions.")

            predictions_df = pd.DataFrame(raw_predictions)
            if predictions_df.shape[1] > 1:
                predictions_df.columns = [f'output_{i}' for i in range(predictions_df.shape[1])]
            else:
                predictions_df.columns = ['prediction']


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
        print(predictions_df.to_string(max_rows=20))
        if predictions_df.shape[0] > 20:
             print(f"... (truncated, {predictions_df.shape[0]} total predictions)")
        print("-------------------\n")
