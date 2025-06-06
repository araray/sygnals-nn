# sygnals_nn/train.py
# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import logging
import numpy as np # Import numpy
from sygnals_nn.utils import load_data
# Import ONNX conversion function if exporting after training
from sygnals_nn.convert import convert_to_onnx


# Configure logging for detailed feedback during execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    export_onnx_path: str | None = None
    ):
    """
    Trains a Keras model using the specified dataset and parameters.

    This function loads a pre-created Keras model, loads training data,
    compiles the model (potentially adjusting the optimizer's learning rate),
    trains the model using model.fit(), saves the updated Keras model,
    and optionally exports the trained model to ONNX format.

    Args:
        model_path (str): Path to the Keras model file (.keras) to load and train.
                          This model should have been created by `sygnals-nn create`.
        data_path (str): Path to the training data file (CSV or JSON).
                         Data should be numerical, suitable for model input.
        epochs (int): Number of complete passes through the training dataset.
        batch_size (int): Number of samples processed in each training step.
        learning_rate (float): The learning rate to use for the optimizer during training.
                               Note: This currently creates a *new* optimizer instance.
        input_cols_str (str): Comma-separated string of input feature column indices or names
                              in the data_path file.
        label_cols_str (str): Comma-separated string of target label column indices or names
                              in the data_path file.
        json_input_key (str): Key name in JSON objects containing the input features
                              (used if data_path is a JSON file). Defaults to 'features'.
        json_label_key (str): Key name in JSON objects containing the target label
                              (used if data_path is a JSON file). Defaults to 'label'.
        export_onnx_path (str | None): If provided, the path where the trained model
                                       will be exported in ONNX format after training.

    Raises:
        FileNotFoundError: If the model or data file does not exist.
        ValueError: If data loading fails, columns are missing, or data shapes mismatch.
        TypeError: If data types are incompatible with model training (e.g., non-numeric labels).
        Exception: For errors during model loading, compilation, training, saving, or ONNX export.
    """
    logging.info(f"Starting training process for model: {model_path}")
    logging.info(f"Training data: {data_path}")
    logging.info(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")

    # --- Validate file paths ---
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data file not found: {data_path}")

    # --- Step 1: Load the Keras Model ---
    try:
        logging.info("Loading Keras model...")
        # Load model without compiling initially. Compilation happens later
        # to ensure the correct learning rate and potentially fix the loss function.
        model = tf.keras.models.load_model(model_path, compile=False)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading Keras model from {model_path}: {e}")
        raise

    # --- Step 2: Load the Training Data ---
    X_train = None # Initialize X_train to handle potential errors in load_data
    Y_train = None # Initialize Y_train
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

        # --- Basic validation of loaded data ---
        if X_train.shape[0] == 0:
             raise ValueError("Loaded training data (X_train) is empty.")
        if Y_train is None:
             # load_data should raise error if labels are missing for training, but double-check
             raise ValueError("Labels (Y_train) could not be loaded for training.")
        if Y_train is not None and X_train.shape[0] != Y_train.shape[0]:
             raise ValueError(f"Number of samples mismatch between features ({X_train.shape[0]}) and labels ({Y_train.shape[0]}).")

        # --- Ensure data types are float32 for training (common requirement for TF/Keras) ---
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
        # Attempt to get original loss and metrics from the loaded model config.
        # This might fail or retrieve incorrect defaults if the model wasn't saved properly
        # or if the loading process doesn't capture compile state well.
        original_loss_func = None
        original_metrics_list = ['accuracy'] # Default metric
        try:
            # Check if the model has compile options stored (Keras sometimes stores this)
             if hasattr(model, 'loss'):
                 original_loss_func = model.loss
             if hasattr(model, 'metrics_names') and len(model.metrics_names) > 1:
                 original_metrics_list = model.metrics_names[1:] # Skip the loss itself

             # Fallback: Try getting from config dict if attributes aren't populated
             if not original_loss_func or not original_metrics_list:
                 config = model.get_config()
                 # Check top-level config and 'compile_config' sub-dict
                 original_loss_func = config.get('loss') or original_loss_func
                 original_metrics_list = config.get('metrics') or original_metrics_list
                 if 'compile_config' in config and config['compile_config']:
                     original_loss_func = config['compile_config'].get('loss') or original_loss_func
                     original_metrics_list = config['compile_config'].get('metrics') or original_metrics_list

        except Exception as config_e:
            logging.warning(f"Could not reliably get original loss/metrics from model config ({config_e}). Will use defaults/overrides.")

        # --- Determine the CORRECT loss function ---
        # Based on the previous error, we know we need sparse_categorical_crossentropy
        # for multi-class classification with integer labels (0, 1, 2...).
        # We will FORCE this loss function here, overriding any potentially incorrect
        # value read from the model file or the previous incorrect default.
        correct_loss_func = 'sparse_categorical_crossentropy'
        logging.info(f"Ensuring loss function is set to '{correct_loss_func}' for multi-class integer labels.")

        # --- Create a new optimizer instance with the specified learning rate ---
        # Note: This approach creates a new optimizer state. If resuming training
        # with the exact previous optimizer state is crucial, loading the saved
        # optimizer and modifying its learning rate is more complex.
        # Creating a new instance is simpler and often sufficient for CLI tools.
        optimizer_name = 'adam' # Default optimizer
        try:
            # Attempt to get the *type* of optimizer from the loaded model to reuse it
            if model.optimizer: # Check if optimizer attribute exists
                # Get class name (e.g., 'Adam', 'SGD') and convert to lowercase string
                optimizer_name = model.optimizer.__class__.__name__.lower()
            elif 'compile_config' in config and config['compile_config']: # Check config again
                 opt_config = config['compile_config'].get('optimizer', {})
                 if isinstance(opt_config, dict):
                      optimizer_name = opt_config.get('class_name', 'adam').lower()
                 elif isinstance(opt_config, str):
                      optimizer_name = opt_config.lower()
            logging.info(f"Attempting to create new '{optimizer_name}' optimizer instance with learning_rate={learning_rate}")
            optimizer_cls = tf.keras.optimizers.get(optimizer_name) # Get class from registry
            new_optimizer = optimizer_cls(learning_rate=learning_rate)
        except Exception as opt_e:
            logging.warning(f"Could not determine original optimizer type or instantiate it ({opt_e}). Using new Adam optimizer.")
            new_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) # Fallback to Adam

        # --- Compile the model with the correct settings ---
        logging.info(f"Compiling model with optimizer='{new_optimizer.__class__.__name__}', loss='{correct_loss_func}', learning_rate={learning_rate}, metrics={original_metrics_list}")
        model.compile(
            optimizer=new_optimizer,
            loss=correct_loss_func, # Use the explicitly correct loss function
            metrics=original_metrics_list # Use original metrics if found, else default ['accuracy']
        )
        logging.info("Model compiled successfully.")
        model.summary(print_fn=logging.info) # Print summary after compilation

    except Exception as e:
        logging.error(f"Error preparing optimizer or compiling model: {e}", exc_info=True)
        raise


    # --- Step 4: Train the Model ---
    try:
        logging.info("Starting model training...")
        # Model should be built if loaded correctly. Functional API models are built
        # on instantiation, Sequential models build on first call or explicit build().
        # The load_model and compile steps should handle this.

        history = model.fit(
            X_train,
            Y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=2 # Print progress bar per epoch (0=silent, 1=bar+metrics, 2=epoch number)
        )
        logging.info("Model training completed.")

        # --- Log final metrics robustly ---
        # Access metrics from the history object
        final_loss = history.history.get('loss', [None])[-1] # Get last loss value
        # Check for common accuracy metric names ('accuracy', 'acc')
        final_accuracy = history.history.get('accuracy', history.history.get('acc', [None]))[-1]

        if final_loss is not None:
             logging.info(f"Final training loss: {final_loss:.4f}")
        if final_accuracy is not None:
            logging.info(f"Final training accuracy: {final_accuracy:.4f}")
        else:
            logging.warning("Could not retrieve final accuracy metric from training history.")

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
            # --- Determine Input Signature for ONNX ---
            # The model should be built now after training. Try inferring from model.inputs.
            input_signature = None
            try:
                # Use model.inputs which should be defined for Functional/built Sequential models
                if hasattr(model, 'inputs') and model.inputs:
                    # Create TensorSpec from each input tensor
                    input_signature = [tf.TensorSpec(shape=inp.shape, dtype=inp.dtype, name=inp.name.split(':')[0])
                                       for inp in model.inputs]
                    logging.info(f"Inferred input signatures for ONNX export from model.inputs: {input_signature}")
                else:
                     # Fallback if .inputs is not available (less common after training)
                     raise ValueError("Could not access model.inputs after training.")

            except Exception as sig_e:
                 logging.warning(f"Could not automatically infer input signature from model attributes ({sig_e}). "
                                 "Attempting signature based on training data shape (may be less reliable).")
                 # Fallback to using the shape of the training data
                 if X_train is not None:
                     try:
                         # Create shape (None, feature1, feature2, ...)
                         inferred_shape = (None,) + X_train.shape[1:]
                         # Try to get dtype from model's input layer, default to float32
                         input_dtype = tf.float32 # Default
                         if hasattr(model, 'input') and hasattr(model.input, 'dtype'):
                              input_dtype = model.input.dtype
                         elif hasattr(model, 'inputs') and model.inputs and hasattr(model.inputs[0], 'dtype'):
                              input_dtype = model.inputs[0].dtype

                         input_signature = [tf.TensorSpec(shape=inferred_shape, dtype=input_dtype)]
                         logging.info(f"Using input signature based on training data shape: {input_signature}")
                     except Exception as shape_e:
                          logging.warning(f"Could not infer signature from data shape: {shape_e}")

            # Proceed with conversion only if a signature was determined
            if not input_signature:
                 logging.error("Failed to determine input signature for ONNX export. Skipping conversion.")
            else:
                 # --- Call the conversion function ---
                 convert_to_onnx(
                     keras_model_path=model_path, # Use the just-saved trained Keras model
                     output_onnx_path=export_onnx_path,
                     input_signature_list=input_signature # Pass the inferred list of tf.TensorSpec
                 )
                 logging.info(f"Model successfully exported to ONNX: {export_onnx_path}")
        except Exception as e:
            # Log the error but don't stop the whole process if Keras saving succeeded
            logging.error(f"Error exporting model to ONNX format: {e}", exc_info=True)
