# sygnals_nn/train.py
# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import logging
import numpy as np # Import numpy
import types # Import the 'types' module for FunctionType
from sygnals_nn.utils import load_data
# Import ONNX conversion function if exporting after training
from sygnals_nn.convert import convert_to_onnx

# Import TensorFlow Probability for probabilistic loss functions
try:
    import tensorflow_probability as tfp
    TFP_AVAILABLE = True
except ImportError:
    TFP_AVAILABLE = False
    # This warning is important for users to understand potential limitations.
    logging.warning("TensorFlow Probability (TFP) not found. Some probabilistic functionalities like NLL losses will not be available.")

# Configure logging for detailed feedback during execution
# This provides insights into the script's operations and helps in debugging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Probabilistic Loss Functions ---
def gaussian_nll_loss(y_true, y_pred_params):
    """
    Negative Log-Likelihood (NLL) loss for a Gaussian distribution.

    This loss function is designed for models that output the parameters of a
    Gaussian distribution, specifically its mean and log-variance (or log-scale).
    It assumes that for each target dimension in `y_true`, the model outputs two values
    in `y_pred_params`: one for the mean and one for the log of the variance.

    Args:
        y_true (tf.Tensor): True target values.
                            Shape: (batch_size, num_target_dims_true).
        y_pred_params (tf.Tensor): Predicted parameters from the model.
                                   Expected to contain at least `2 * num_target_dims_true` elements
                                   along the last axis, ordered as [means..., log_variances...].
                                   Shape: (batch_size, >=2 * num_target_dims_true).
                                   If larger, only the initial required parameters are used.

    Returns:
        tf.Tensor: The mean negative log-likelihood loss over the batch.
                   A scalar tensor representing the average loss per sample.

    Raises:
        ImportError: If TensorFlow Probability is not installed.
        tf.errors.InvalidArgumentError: If `y_pred_params` last dimension is less than `2 * num_target_dims_true`.
    """
    if not TFP_AVAILABLE:
        raise ImportError("TensorFlow Probability is required for gaussian_nll_loss. Please install it.")

    # Determine the number of target dimensions from the shape of y_true
    num_target_dims_true = tf.shape(y_true)[-1]

    # Ensure y_pred_params has enough components for the means and log_variances
    # for the number of dimensions in y_true.
    expected_pred_params_dim = 2 * num_target_dims_true

    # Check if y_pred_params has enough elements. This check will run in graph mode.
    tf.Assert(tf.greater_equal(tf.shape(y_pred_params)[-1], expected_pred_params_dim),
              [f"y_pred_params last dimension must be at least 2 * num_target_dims_true ({expected_pred_params_dim}),",
               "but got shape:", tf.shape(y_pred_params)])

    # Slice y_pred_params to get the mean and log_variance components.
    # Only use the first num_target_dims_true for means and the next num_target_dims_true for log_variances.
    mean = y_pred_params[..., :num_target_dims_true]
    log_variance = y_pred_params[..., num_target_dims_true:expected_pred_params_dim]

    # Calculate the scale (standard deviation) from log_variance.
    scale = tf.exp(log_variance * 0.5)

    # Create a Normal distribution object.
    # loc and scale will both have shape (batch_size, num_target_dims_true)
    dist = tfp.distributions.Normal(loc=mean, scale=scale)

    # Calculate the negative log probability (NLL).
    # nll will have shape (batch_size, num_target_dims_true).
    nll = -dist.log_prob(y_true)

    # Use tf.cond for conditional logic based on num_target_dims_true.
    def sum_nll_over_dimensions():
        # If multiple target dimensions, sum NLL across these dimensions for each sample.
        return tf.reduce_sum(nll, axis=-1) # Result shape: (batch_size,)

    def pass_through_nll():
        # If num_target_dims_true is 1, nll is (batch_size, 1). Squeeze to (batch_size,).
        # This ensures the output of tf.cond has a consistent rank.
        return tf.squeeze(nll, axis=-1) # Result shape: (batch_size,)

    # Conditionally process nll.
    # The result (processed_nll) will have shape (batch_size,).
    processed_nll = tf.cond(
        tf.greater(num_target_dims_true, 1),
        true_fn=sum_nll_over_dimensions,
        false_fn=pass_through_nll # Renamed for clarity, still squeezes
    )

    # Return the mean NLL over all samples in the batch.
    return tf.reduce_mean(processed_nll)


# Mapping of string names to custom (and potentially standard) loss functions
CUSTOM_LOSS_MAP = {
    "gaussian_nll": gaussian_nll_loss,
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

    This function loads a pre-created Keras model, loads training data,
    compiles the model (potentially adjusting the optimizer's learning rate and
    the loss function if overridden), trains the model using `model.fit()`,
    saves the updated Keras model, and optionally exports the trained model
    to ONNX format.

    The training pipeline involves these main steps:
    1.  **File Validation**: Checks if the model and data files exist. This is crucial
        for preventing errors later in the process.
    2.  **Model Loading**: Loads the Keras model structure from `model_path`.
        The model is loaded with `compile=False` initially. This is a key step because
        it allows the optimizer's learning rate (and potentially the loss function itself,
        if overridden) to be set specifically for the current training session, rather
        than being fixed by how the model was last saved.
    3.  **Data Loading**: Loads features (X_train) and labels (Y_train) from
        `data_path` (which can be a CSV or JSON file) using the specified
        column or key identifiers. The utility function `load_data` handles the
        parsing and ensures that the data is converted to `float32`, a common
        requirement for TensorFlow/Keras models.
    4.  **Loss Function Determination**: This is a critical step, especially for
        flexibility and supporting probabilistic models.
        * If `training_loss_override` is provided by the user (e.g., 'gaussian_nll'
            for a probabilistic regression model), this specified loss function is used.
            It can be a key from the `CUSTOM_LOSS_MAP` (for custom NLL losses) or a
            standard Keras loss identifier string (e.g., 'binary_crossentropy').
        * If no override is given (`training_loss_override` is None), the function
            attempts to retrieve the loss function that the Keras model was originally
            configured with. This involves inspecting internal attributes of the loaded
            model where Keras stores compilation arguments (like `_compile_config` for
            `.keras` v3 format or `_training_config` for older H5 formats). This ensures
            that if a model was created with, for example, 'sparse_categorical_crossentropy',
            that same loss is used for training unless explicitly changed.
    5.  **Optimizer Preparation**: A new optimizer instance is created for each training
        run. The type of optimizer (e.g., Adam, SGD) is inferred from the loaded model's
        configuration if possible; otherwise, Adam is used as a default. The crucial part
        is that this new instance is configured with the `learning_rate` provided as an
        argument to this function. This means the user has direct control over the
        learning rate for the current training session, and the optimizer's state (like
        moments in Adam) is reinitialized.
    6.  **Model Compilation**: The loaded Keras model is then compiled using the
        determined loss function and the newly created optimizer instance. Standard
        metrics (like 'mae', 'mse', and 'accuracy' if the task appears to be
        classification based on the loss or label data) are typically included to monitor
        training progress.
    7.  **Model Training**: The actual training is performed by calling `model.fit()`
        with the prepared `X_train`, `Y_train`, and the specified number of `epochs`
        and `batch_size`. Training progress is logged.
    8.  **Model Saving**: After training completes, the updated Keras model (which now
        includes the learned weights and the state of the optimizer used for this
        training session) is saved back to the original `model_path`, overwriting it.
    9.  **ONNX Export (Optional)**: If `export_onnx_path` is specified, the newly
        trained Keras model is converted to the ONNX format and saved to the given
        path. This allows for broader deployment options.

    Args:
        model_path (str): Path to the Keras model file (.keras) to load and train.
                          This model should have its architecture defined, typically by `sygnals-nn create`.
        data_path (str): Path to the training data file (CSV or JSON format).
                         Data should be numerical or appropriately preprocessed for model input.
        epochs (int): Number of complete passes through the entire training dataset.
        batch_size (int): Number of samples processed in each training iteration (batch).
        learning_rate (float): The learning rate for the optimizer. A new optimizer instance
                               is created with this learning rate for each training run.
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
                                             to use the loss from the model's saved configuration.
                                             Defaults to None.

    Raises:
        FileNotFoundError: If the `model_path` or `data_path` does not exist.
        ValueError: If data loading fails (e.g., missing columns, incorrect format),
                    if data shapes are inconsistent, if a specified
                    `training_loss_override` is not recognized, or if the loss function
                    cannot be determined from the model's configuration and no override
                    is provided.
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

    # --- Step 1: Validate file paths ---
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data file not found: {data_path}")

    # --- Step 2: Load the Keras Model ---
    try:
        logging.info("Loading Keras model from: %s", model_path)
        # Load model without compiling initially. This allows us to set a new learning rate
        # for the optimizer or even change the loss function if overridden.
        model = tf.keras.models.load_model(model_path, compile=False)
        logging.info("Keras model loaded successfully.")
        # Retrieve the model's architecture configuration. This might be used to infer
        # the original optimizer type or loss if not overridden.
        model_architecture_config = model.get_config()
    except Exception as e:
        logging.error(f"Error loading Keras model from {model_path}: {e}")
        raise

    # --- Step 3: Load the Training Data ---
    X_train, Y_train = None, None # Initialize to handle potential errors in load_data
    try:
        logging.info("Loading training data from: %s", data_path)
        # Use the utility function to load data from CSV or JSON
        X_train, Y_train = load_data(
            file_path=data_path,
            input_cols_str=input_cols_str,
            label_cols_str=label_cols_str,
            json_input_key=json_input_key,
            json_label_key=json_label_key,
            is_inference=False # Explicitly False for training, ensuring labels are loaded
        )
        logging.info(f"Training data loaded. X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape if Y_train is not None else 'None'}")

        # Basic validation of loaded data
        if X_train.shape[0] == 0:
             raise ValueError("Loaded training data (X_train) is empty.")
        if Y_train is None: # Should be caught by load_data if is_inference=False, but good to check
             raise ValueError("Labels (Y_train) could not be loaded for training. Ensure `is_inference` is False in load_data if called directly.")
        if X_train.shape[0] != Y_train.shape[0]: # Ensure features and labels have the same number of samples
             raise ValueError(f"Number of samples mismatch between features ({X_train.shape[0]}) and labels ({Y_train.shape[0]}).")

        # Ensure data types are float32 for training, as commonly expected by TensorFlow/Keras
        if X_train.dtype != np.float32:
            logging.warning(f"Converting X_train dtype from {X_train.dtype} to float32.")
            X_train = X_train.astype(np.float32)
        if Y_train.dtype != np.float32:
            # Before casting labels, ensure they are numeric to avoid errors
            if not np.issubdtype(Y_train.dtype, np.number):
                 raise TypeError(f"Loaded labels (Y_train) have non-numeric dtype {Y_train.dtype} and cannot be cast to float32 directly.")
            logging.warning(f"Converting Y_train dtype from {Y_train.dtype} to float32.")
            Y_train = Y_train.astype(np.float32)
    except Exception as e:
        logging.error(f"Error loading or processing training data from {data_path}: {e}")
        raise

    # --- Step 4 & 5: Prepare Optimizer and Loss, then Compile ---
    try:
        loss_to_use = None
        # Priority 1: Use training_loss_override if provided by the user
        if training_loss_override:
            if training_loss_override in CUSTOM_LOSS_MAP: # Check our custom loss map first
                loss_to_use = CUSTOM_LOSS_MAP[training_loss_override]
                logging.info(f"Using custom mapped loss function from override: '{training_loss_override}' (Function: {loss_to_use.__name__})")
            else:
                try:
                    # Attempt to get it as a standard Keras loss identifier (string or function)
                    loss_to_use = tf.keras.losses.get(training_loss_override)
                    logging.info(f"Using Keras loss function from override: '{training_loss_override}'")
                except ValueError: # Keras raises ValueError if identifier is unknown
                    raise ValueError(f"Specified training_loss_override '{training_loss_override}' is not a recognized custom mapped loss or standard Keras loss identifier.")
        else:
            # Priority 2: No override, try to get loss from the loaded model's internal saved configuration.
            # Keras models save their compile arguments. When loaded with compile=False, these are often in protected attributes.
            retrieved_loss_config = None
            # For Keras v3 format (.keras zip files), compile arguments are typically in _compile_config
            if hasattr(model, '_compile_config') and model._compile_config and isinstance(model._compile_config, dict):
                retrieved_loss_config = model._compile_config.get('loss')
                if retrieved_loss_config:
                    logging.info(f"Retrieved loss configuration from model._compile_config (Keras v3 format): {retrieved_loss_config}")

            # For older H5 format, compile arguments might be in _training_config
            if not retrieved_loss_config and hasattr(model, '_training_config') and model._training_config and isinstance(model._training_config, dict):
                retrieved_loss_config = model._training_config.get('loss')
                if retrieved_loss_config:
                    logging.info(f"Retrieved loss configuration from model._training_config (H5 format): {retrieved_loss_config}")

            # Fallback: Check the general model architecture config (less reliable for compile-time loss when loaded with compile=False)
            if not retrieved_loss_config and model_architecture_config:
                loss_in_main_config = model_architecture_config.get('loss')
                if loss_in_main_config:
                    retrieved_loss_config = loss_in_main_config
                    logging.info(f"Retrieved loss configuration from model.get_config().get('loss') (fallback): {retrieved_loss_config}")

            if retrieved_loss_config:
                try:
                    # tf.keras.losses.get can deserialize string names or configuration dictionaries
                    loss_to_use = tf.keras.losses.get(retrieved_loss_config)
                    # For logging, get a representative name of the loss
                    current_loss_name_log = retrieved_loss_config if isinstance(retrieved_loss_config, str) \
                                           else loss_to_use.__class__.__name__ if hasattr(loss_to_use, '__class__') and not isinstance(loss_to_use, types.FunctionType) \
                                           else loss_to_use.__name__ if callable(loss_to_use) \
                                           else str(retrieved_loss_config) # Fallback to string of config
                    logging.info(f"Using loss from model's saved configuration: '{current_loss_name_log}'")
                except Exception as e_get_loss:
                    raise ValueError(f"Failed to interpret loss from model's saved configuration ('{retrieved_loss_config}'). Please specify a valid --training-loss. Original error: {e_get_loss}")
            else:
                # This path means no loss override and no loss found in model's typical config locations.
                raise ValueError("Loss function is undefined in the loaded model and not specified via --training-loss. Please provide a loss function for training.")

        optimizer_name = 'adam' # Default optimizer
        # Try to get optimizer type from model config to maintain consistency
        if hasattr(model, '_compile_config') and model._compile_config and isinstance(model._compile_config.get('optimizer'), dict):
            optimizer_config_dict = model._compile_config.get('optimizer')
            optimizer_name = optimizer_config_dict.get('class_name', optimizer_name).lower()
        elif hasattr(model, '_training_config') and model._training_config and isinstance(model._training_config.get('optimizer_config'), dict):
            optimizer_config_dict = model._training_config.get('optimizer_config')
            optimizer_name = optimizer_config_dict.get('class_name', optimizer_name).lower()
        elif model_architecture_config: # Fallback checks
            compile_config_keras3 = model_architecture_config.get('compile_config')
            if compile_config_keras3 and isinstance(compile_config_keras3.get('optimizer'), dict):
                optimizer_name = compile_config_keras3.get('optimizer').get('class_name', optimizer_name).lower()
            elif isinstance(model_architecture_config.get('optimizer'), dict):
                 optimizer_name = model_architecture_config['optimizer'].get('class_name', optimizer_name).lower()

        logging.info(f"Attempting to create new '{optimizer_name}' optimizer instance with learning_rate={learning_rate}")
        try:
            optimizer_cls = tf.keras.optimizers.get(optimizer_name)
            new_optimizer = optimizer_cls(learning_rate=learning_rate)
        except Exception as opt_e:
            logging.warning(f"Could not determine original optimizer type from config ('{optimizer_name}', error: {opt_e}). Using new Adam optimizer as fallback.")
            new_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        current_loss_name_display = loss_to_use if isinstance(loss_to_use, str) \
                                   else loss_to_use.__class__.__name__ if hasattr(loss_to_use, '__class__') and not isinstance(loss_to_use, types.FunctionType) \
                                   else loss_to_use.__name__ if callable(loss_to_use) \
                                   else str(loss_to_use)
        logging.info(f"Compiling model with optimizer='{new_optimizer.__class__.__name__}', loss='{current_loss_name_display}', learning_rate={learning_rate}")

        metrics_to_use = ['mae', 'mse']
        # Heuristic for classification tasks
        is_classification_loss = 'categorical_crossentropy' in str(current_loss_name_display).lower() or \
                                 'binary_crossentropy' in str(current_loss_name_display).lower()
        is_classification_data = (Y_train.ndim == 2 and Y_train.shape[1] > 1 and np.all((Y_train == 0) | (Y_train == 1))) or \
                                 (Y_train.ndim == 1 and len(np.unique(Y_train)) < 20 and np.issubdtype(Y_train.dtype, np.integer))
        if is_classification_loss or is_classification_data:
            if 'accuracy' not in metrics_to_use: metrics_to_use.append('accuracy')

        model.compile(optimizer=new_optimizer, loss=loss_to_use, metrics=metrics_to_use)
        logging.info("Model compiled successfully.")
        model.summary(print_fn=logging.info)

    except Exception as e:
        logging.error(f"Error preparing optimizer or compiling model: {e}", exc_info=True)
        raise

    # --- Step 6: Train the Model ---
    try:
        logging.info("Starting model training...")
        history = model.fit(
            X_train, Y_train,
            epochs=epochs, batch_size=batch_size, verbose=2
        )
        logging.info("Model training completed.")
        final_loss = history.history.get('loss', [None])[-1]
        log_msg = f"Final training metrics: Loss: {final_loss:.4f}"
        for metric_name in model.metrics_names:
            if metric_name != 'loss' and metric_name in history.history:
                final_metric_val = history.history[metric_name][-1]
                log_msg += f", {metric_name.capitalize()}: {final_metric_val:.4f}"
        logging.info(log_msg)
    except Exception as e:
        logging.error(f"Error during model training: {e}", exc_info=True)
        raise

    # --- Step 7: Save the Trained Keras Model ---
    try:
        output_dir = os.path.dirname(model_path)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        model.save(model_path)
        logging.info(f"Trained Keras model saved successfully to {model_path}")
    except Exception as e:
        logging.error(f"Error saving trained Keras model to {model_path}: {e}")
        raise

    # --- Step 8: Optional ONNX Export ---
    if export_onnx_path:
        logging.info(f"Exporting trained model to ONNX format: {export_onnx_path}")
        try:
            input_signature = None
            if hasattr(model, 'inputs') and model.inputs:
                input_signature = [tf.TensorSpec(shape=inp.shape, dtype=inp.dtype, name=inp.name.split(':')[0])
                                   for inp in model.inputs]
            elif X_train is not None: # Fallback to training data
                inferred_shape = (None,) + X_train.shape[1:]
                input_dtype = X_train.dtype # Default to X_train's dtype
                # Try to get a more specific dtype from model's input layer if possible
                if hasattr(model, 'input') and hasattr(model.input, 'dtype'): # Keras 2
                     input_dtype = model.input.dtype
                elif hasattr(model, 'inputs') and model.inputs and hasattr(model.inputs[0], 'dtype'): # Keras 3
                     input_dtype = model.inputs[0].dtype
                input_signature = [tf.TensorSpec(shape=inferred_shape, dtype=input_dtype)]

            if not input_signature:
                 logging.error("Failed to determine input signature for ONNX export. Skipping conversion.")
            else:
                 logging.info(f"Using input signature for ONNX export: {input_signature}")
                 convert_to_onnx(
                     keras_model_path=model_path,
                     output_onnx_path=export_onnx_path,
                     input_signature_list=input_signature
                 )
                 logging.info(f"Model successfully exported to ONNX: {export_onnx_path}")
        except Exception as e:
            logging.error(f"Error exporting model to ONNX format: {e}", exc_info=True)
