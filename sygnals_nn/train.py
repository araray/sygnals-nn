# sygnals_nn/train.py
# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import logging
import numpy as np # Import numpy
from sygnals_nn.utils import load_data
# Import ONNX conversion function if exporting after training
from sygnals_nn.convert import convert_to_onnx

# Import TensorFlow Probability for probabilistic loss functions
try:
    import tensorflow_probability as tfp
    TFP_AVAILABLE = True
except ImportError:
    TFP_AVAILABLE = False
    logging.warning("TensorFlow Probability (TFP) not found. Some probabilistic functionalities like NLL losses will not be available.")

# Configure logging for detailed feedback during execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Probabilistic Loss Functions ---
def gaussian_nll_loss(y_true, y_pred_params):
    """
    Negative Log-Likelihood (NLL) loss for a Gaussian distribution.

    This loss function is designed for models that output the parameters of a
    Gaussian distribution, specifically its mean and log-variance (or log-scale).
    It assumes that for each target dimension, the model outputs two values:
    one for the mean and one for the log of the variance.

    Args:
        y_true (tf.Tensor): True target values.
                            Shape: (batch_size, num_target_dimensions).
        y_pred_params (tf.Tensor): Predicted parameters from the model.
                                   Shape: (batch_size, 2 * num_target_dimensions).
                                   The parameters are assumed to be ordered such that
                                   the first `num_target_dimensions` elements are the means,
                                   and the next `num_target_dimensions` elements are the log-variances.
                                   For a single target dimension (e.g., univariate regression),
                                   this would be [mean, log_variance].

    Returns:
        tf.Tensor: The mean negative log-likelihood loss over the batch.
                   A scalar tensor representing the average loss per sample.

    Raises:
        ImportError: If TensorFlow Probability is not installed, as it's
                     required for defining the Normal distribution.
    """
    if not TFP_AVAILABLE:
        raise ImportError("TensorFlow Probability is required for gaussian_nll_loss. Please install it.")

    # Determine the number of target dimensions from the shape of y_true
    num_target_dimensions = tf.shape(y_true)[-1] # Gets the last dimension size

    # Split the predicted parameters into mean and log_variance components
    # Example: if y_pred_params is (batch, D+D), mean is (batch, D), log_variance is (batch, D)
    mean = y_pred_params[..., :num_target_dimensions]
    log_variance = y_pred_params[..., num_target_dimensions:]

    # Calculate the scale (standard deviation) from log_variance.
    # scale = sqrt(variance) = sqrt(exp(log_variance)) = exp(0.5 * log_variance).
    # This ensures the scale is always positive.
    scale = tf.exp(log_variance * 0.5)

    # Create a Normal distribution object using the predicted mean and calculated scale.
    # `tfp.distributions.Normal` creates a distribution instance.
    # `loc` is the mean, `scale` is the standard deviation.
    dist = tfp.distributions.Normal(loc=mean, scale=scale)

    # Calculate the negative log probability (NLL) of the true values under the predicted distribution.
    # `dist.log_prob(y_true)` computes log P(y_true | mean, scale).
    # We take the negative of this for minimization.
    # The shape of nll will be (batch_size, num_target_dimensions).
    nll = -dist.log_prob(y_true)

    # If there are multiple target dimensions, sum the NLL across these dimensions for each sample.
    # This assumes independence of the target variables given the input, a common assumption.
    if num_target_dimensions > 1:
        nll = tf.reduce_sum(nll, axis=-1) # Sum over the last axis (dimensions)
                                          # Shape becomes (batch_size,)

    # Return the mean NLL over all samples in the batch.
    return tf.reduce_mean(nll)


# Mapping of string names to custom (and potentially standard) loss functions
# This allows users to specify losses by name via CLI or configuration.
CUSTOM_LOSS_MAP = {
    "gaussian_nll": gaussian_nll_loss,
    # Future NLL losses (e.g., "laplace_nll", "student_t_nll") can be added here.
}


def train_model(
    model_path: str,
    data_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    input_cols_str: str,
    label_cols_str: str,
    json_input_key: str = 'features',
    json_label_key: str = 'label',
    export_onnx_path: str | None = None,
    training_loss_override: str | None = None
    ):
    """
    Trains a Keras model using the specified dataset and parameters.

    This function orchestrates the model training pipeline:
    1.  Loads a pre-created Keras model structure (from `sygnals-nn create`).
    2.  Loads and preprocesses the training data (features X_train, labels Y_train).
    3.  Determines the appropriate loss function:
        * If `training_loss_override` is specified (e.g., 'gaussian_nll'), that loss is used.
            This is particularly useful for probabilistic models requiring custom NLL losses.
        * Otherwise, it attempts to use the loss function the model was last compiled with
            or the one defined in its configuration during the `create` step.
    4.  Creates a new optimizer instance (e.g., Adam) with the specified `learning_rate`.
    5.  Compiles the Keras model with the chosen optimizer and loss function.
    6.  Trains the model using `model.fit()` with the provided data, epochs, and batch size.
    7.  Saves the updated (trained) Keras model back to `model_path`.
    8.  Optionally, if `export_onnx_path` is provided, converts the trained Keras model
        to ONNX format and saves it.

    Args:
        model_path (str): Path to the Keras model file (.keras) to load and train.
                          This model should have its architecture defined, typically by `sygnals-nn create`.
        data_path (str): Path to the training data file (CSV or JSON format).
                         Data should be numerical or appropriately preprocessed for model input.
        epochs (int): Number of complete passes through the entire training dataset.
        batch_size (int): Number of samples processed in each training iteration (batch).
        learning_rate (float): The learning rate for the optimizer. A new optimizer instance
                               is created with this learning rate.
        input_cols_str (str): Comma-separated string of column indices or names in `data_path`
                              to be used as input features (X_train).
        label_cols_str (str): Comma-separated string of column indices or names in `data_path`
                              to be used as target labels (Y_train).
        json_input_key (str): Key name in JSON objects that contains the input features.
                              Used if `data_path` is a JSON file. Defaults to 'features'.
        json_label_key (str): Key name in JSON objects that contains the target label.
                              Used if `data_path` is a JSON file. Defaults to 'label'.
        export_onnx_path (str | None): If provided, the path where the trained Keras model
                                       will be exported to ONNX format after training.
                                       Defaults to None (no ONNX export).
        training_loss_override (str | None): Optional string key to specify a particular loss
                                             function for this training run. This is useful for
                                             probabilistic models (e.g., 'gaussian_nll') or if
                                             the loss needs to be different from the one saved
                                             in the Keras model file. If None, the function attempts
                                             to use the loss from the model's configuration.
                                             Defaults to None.

    Raises:
        FileNotFoundError: If the `model_path` or `data_path` does not exist.
        ValueError: If data loading fails (e.g., missing columns, incorrect format),
                    if data shapes are inconsistent, or if a specified
                    `training_loss_override` is not recognized.
        TypeError: If loaded data (features or labels) has a non-numeric dtype that
                   cannot be converted to float32 for training.
        ImportError: If TensorFlow Probability is required for a specified loss function
                     (like 'gaussian_nll') but is not installed.
        Exception: For other errors during model loading, compilation, training,
                   saving, or ONNX conversion (e.g., TensorFlow internal errors).
    """
    logging.info(f"Starting training process for model: {model_path}")
    logging.info(f"Training data: {data_path}")
    logging.info(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")
    if training_loss_override:
        logging.info(f"Training loss override specified: '{training_loss_override}'")

    # --- Validate file paths ---
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data file not found: {data_path}")

    # --- Step 1: Load the Keras Model ---
    try:
        logging.info("Loading Keras model...")
        # Load model without compiling initially. Compilation happens later
        # to ensure the correct learning rate and loss function are applied.
        model = tf.keras.models.load_model(model_path, compile=False)
        logging.info("Model loaded successfully.")
        # Store model's original config for later reference if needed for optimizer or loss.
        model_config = model.get_config()
    except Exception as e:
        logging.error(f"Error loading Keras model from {model_path}: {e}")
        raise

    # --- Step 2: Load the Training Data ---
    X_train, Y_train = None, None # Initialize to handle potential errors in load_data
    try:
        logging.info("Loading training data...")
        # Use the utility function to load data from CSV or JSON
        X_train, Y_train = load_data(
            file_path=data_path,
            input_cols_str=input_cols_str,
            label_cols_str=label_cols_str,
            json_input_key=json_input_key,
            json_label_key=json_label_key,
            is_inference=False # Explicitly False for training
        )
        logging.info(f"Training data loaded. X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape if Y_train is not None else 'None'}")

        # Basic validation of loaded data
        if X_train.shape[0] == 0:
             raise ValueError("Loaded training data (X_train) is empty.")
        if Y_train is None:
             # load_data should raise error if labels are missing for training, but double-check
             raise ValueError("Labels (Y_train) could not be loaded for training.")
        if Y_train is not None and X_train.shape[0] != Y_train.shape[0]:
             raise ValueError(f"Number of samples mismatch between features ({X_train.shape[0]}) and labels ({Y_train.shape[0]}).")

        # Ensure data types are float32 for training (common requirement for TF/Keras)
        if X_train.dtype != np.float32:
            logging.warning(f"Converting X_train dtype from {X_train.dtype} to float32.")
            X_train = X_train.astype(np.float32)
        if Y_train is not None and Y_train.dtype != np.float32:
            # Check if Y_train is already numeric before casting to avoid errors
            if not np.issubdtype(Y_train.dtype, np.number):
                 # This should have been caught by load_data, but safeguard here
                 raise TypeError(f"Loaded labels (Y_train) have non-numeric dtype {Y_train.dtype} before casting.")
            logging.warning(f"Converting Y_train dtype from {Y_train.dtype} to float32.")
            Y_train = Y_train.astype(np.float32)

    except Exception as e:
        logging.error(f"Error loading training data from {data_path}: {e}")
        raise

    # --- Step 3: Prepare Optimizer and Compile ---
    try:
        # Determine the loss function to use for this training session
        loss_to_use = None
        if training_loss_override:
            if training_loss_override in CUSTOM_LOSS_MAP:
                loss_to_use = CUSTOM_LOSS_MAP[training_loss_override]
                logging.info(f"Using custom mapped loss function: '{training_loss_override}' ({loss_to_use.__name__})")
            elif training_loss_override in tf.keras.losses.get.__globals__ or hasattr(tf.keras.losses, training_loss_override):
                # Check if it's a standard Keras loss name
                loss_to_use = training_loss_override
                logging.info(f"Using standard Keras loss function: '{training_loss_override}'")
            else:
                # If not found in custom map or standard Keras losses by string name
                raise ValueError(f"Specified training_loss_override '{training_loss_override}' is not a recognized custom loss or standard Keras loss name.")
        else:
            # Fallback to loss from model's compile history or config if no override
            try:
                # model.loss might be populated if model was saved after compile, or from config
                loss_from_config = model.loss if hasattr(model, 'loss') and model.loss else model_config.get('loss', 'auto')
                if isinstance(loss_from_config, dict): # Keras sometimes stores loss as dict (e.g. {'class_name': 'BinaryCrossentropy', ...})
                    loss_to_use = loss_from_config.get('class_name', 'auto').lower()
                elif isinstance(loss_from_config, str):
                    loss_to_use = loss_from_config
                else: # Could be a function object if loaded with custom objects
                    loss_to_use = loss_from_config

                logging.info(f"Using loss from model configuration or default: '{loss_to_use if isinstance(loss_to_use, str) else loss_to_use.__name__ if callable(loss_to_use) else str(loss_to_use)}'")

            except Exception as e_loss_config:
                 logging.warning(f"Could not reliably determine loss from model config ({e_loss_config}). Defaulting to 'auto' for TensorFlow if not overridden.")
                 loss_to_use = 'auto' # 'auto' lets TensorFlow decide based on output shape/activation

        # Prepare Optimizer: Create a new instance with the specified learning rate
        optimizer_name = 'adam' # Default optimizer
        try:
            # Attempt to get the *type* of optimizer from the loaded model to reuse its class
            if hasattr(model, 'optimizer') and model.optimizer:
                optimizer_name = model.optimizer.__class__.__name__.lower()
            elif 'optimizer_config' in model_config and isinstance(model_config.get('optimizer_config'), dict) : # Keras 3.x style
                 optimizer_name = model_config['optimizer_config'].get('class_name', 'adam').lower()
            elif 'optimizer' in model_config and isinstance(model_config.get('optimizer'), dict): # Older Keras style
                 optimizer_name = model_config['optimizer'].get('class_name', 'adam').lower()

            logging.info(f"Attempting to create new '{optimizer_name}' optimizer instance with learning_rate={learning_rate}")
            optimizer_cls = tf.keras.optimizers.get(optimizer_name) # Get class from Keras registry
            new_optimizer = optimizer_cls(learning_rate=learning_rate)
        except Exception as opt_e:
            logging.warning(f"Could not determine original optimizer type or instantiate it ({opt_e}). Using new Adam optimizer as fallback.")
            new_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Compile the model with the determined optimizer and loss
        current_loss_name = loss_to_use if isinstance(loss_to_use, str) else loss_to_use.__name__ if callable(loss_to_use) else str(loss_to_use)
        logging.info(f"Compiling model with optimizer='{new_optimizer.__class__.__name__}', loss='{current_loss_name}', learning_rate={learning_rate}")

        # Default metrics; can be enhanced later
        metrics_to_use = ['mae', 'mse'] # Good defaults for regression
        if 'categorical_crossentropy' in str(current_loss_name).lower() or \
           'binary_crossentropy' in str(current_loss_name).lower() or \
           (Y_train.ndim == 2 and Y_train.shape[1] > 1 and np.all((Y_train == 0) | (Y_train == 1))): # Likely classification
            metrics_to_use.append('accuracy')

        model.compile(
            optimizer=new_optimizer,
            loss=loss_to_use,
            metrics=metrics_to_use
        )
        logging.info("Model compiled successfully.")
        model.summary(print_fn=logging.info) # Print model summary after compilation

    except Exception as e:
        logging.error(f"Error preparing optimizer or compiling model: {e}", exc_info=True)
        raise

    # --- Step 4: Train the Model ---
    try:
        logging.info("Starting model training...")
        history = model.fit(
            X_train,
            Y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=2 # Print progress bar per epoch (0=silent, 1=bar+metrics, 2=epoch number)
        )
        logging.info("Model training completed.")

        # Log final metrics from the training history
        final_loss = history.history.get('loss', [None])[-1]
        log_msg = "Final training metrics: "
        if final_loss is not None: log_msg += f"Loss: {final_loss:.4f}"

        # Log other metrics that were compiled
        for metric_name in model.metrics_names:
            if metric_name != 'loss' and metric_name in history.history:
                final_metric_val = history.history[metric_name][-1]
                log_msg += f", {metric_name.capitalize()}: {final_metric_val:.4f}"
        logging.info(log_msg)

    except Exception as e:
        logging.error(f"Error during model training: {e}", exc_info=True)
        raise

    # --- Step 5: Save the Trained Keras Model ---
    # Overwrite the original model file with the newly trained weights and state
    try:
        output_dir = os.path.dirname(model_path)
        if output_dir: # Only create if path includes a directory
             os.makedirs(output_dir, exist_ok=True)
        # Save the entire model (architecture, weights, optimizer state)
        model.save(model_path)
        logging.info(f"Trained Keras model saved successfully to {model_path}")
    except Exception as e:
        logging.error(f"Error saving trained Keras model to {model_path}: {e}")
        raise # Re-raise the error after logging

    # --- Step 6: Optional ONNX Export ---
    if export_onnx_path:
        logging.info(f"Exporting trained model to ONNX format: {export_onnx_path}")
        try:
            # Determine Input Signature for ONNX conversion
            input_signature = None
            if hasattr(model, 'inputs') and model.inputs: # Preferred way for built models
                input_signature = [tf.TensorSpec(shape=inp.shape, dtype=inp.dtype, name=inp.name.split(':')[0])
                                   for inp in model.inputs]
                logging.info(f"Inferred input signatures for ONNX export from model.inputs: {input_signature}")
            elif X_train is not None: # Fallback to using the shape of the training data
                try:
                    inferred_shape = (None,) + X_train.shape[1:] # Add None for batch dimension
                    # Try to get dtype from model's input layer, default to X_train's dtype or float32
                    input_dtype = X_train.dtype if hasattr(X_train, 'dtype') else tf.float32
                    if hasattr(model, 'input') and hasattr(model.input, 'dtype'):
                         input_dtype = model.input.dtype
                    elif hasattr(model, 'inputs') and model.inputs and hasattr(model.inputs[0], 'dtype'):
                         input_dtype = model.inputs[0].dtype
                    input_signature = [tf.TensorSpec(shape=inferred_shape, dtype=input_dtype)]
                    logging.info(f"Using input signature based on training data shape for ONNX: {input_signature}")
                except Exception as shape_e:
                     logging.warning(f"Could not infer signature from training data shape: {shape_e}")

            if not input_signature:
                 logging.error("Failed to determine input signature for ONNX export. Skipping conversion.")
            else:
                 # Call the conversion function
                 convert_to_onnx(
                     keras_model_path=model_path, # Use the just-saved trained Keras model
                     output_onnx_path=export_onnx_path,
                     input_signature_list=input_signature # Pass the inferred list of tf.TensorSpec
                 )
                 logging.info(f"Model successfully exported to ONNX: {export_onnx_path}")
        except Exception as e:
            # Log the error but don't stop the whole process if Keras saving succeeded
            logging.error(f"Error exporting model to ONNX format: {e}", exc_info=True)
