# sygnals_nn/train.py
# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import logging
from sygnals_nn.utils import load_data
# Import ONNX conversion function if exporting after training
from sygnals_nn.convert import convert_to_onnx
import numpy as np # Import numpy

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

    Args:
        model_path: Path to the Keras model file (.keras) to load and train.
        data_path: Path to the training data file (CSV or JSON).
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for the optimizer.
        input_cols_str: Comma-separated input column indices or names.
        label_cols_str: Comma-separated label column indices or names.
        json_input_key: Key for input features in JSON data.
        json_label_key: Key for label in JSON data.
        export_onnx_path: If provided, path to save the trained model in ONNX format.

    Raises:
        FileNotFoundError: If the model or data file does not exist.
        Exception: For errors during loading, training, or saving.
    """
    logging.info(f"Starting training process for model: {model_path}")
    logging.info(f"Training data: {data_path}")
    logging.info(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data file not found: {data_path}")

    # --- Step 1: Load the Keras Model ---
    try:
        logging.info("Loading Keras model...")
        # Load model without compiling initially to inspect optimizer later if needed
        model = tf.keras.models.load_model(model_path, compile=False)
        logging.info("Model loaded successfully.")
        # We'll compile later after potentially modifying the optimizer
        # model.summary(print_fn=logging.info) # Print summary after compilation
    except Exception as e:
        logging.error(f"Error loading Keras model from {model_path}: {e}")
        raise

    # --- Step 2: Load the Training Data ---
    X_train = None # Initialize X_train to handle potential errors in load_data
    Y_train = None # Initialize Y_train
    try:
        logging.info("Loading training data...")
        X_train, Y_train = load_data(
            file_path=data_path,
            input_cols_str=input_cols_str,
            label_cols_str=label_cols_str,
            json_input_key=json_input_key,
            json_label_key=json_label_key,
            is_inference=False # Explicitly False for training
        )
        logging.info(f"Training data loaded. X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape if Y_train is not None else 'None'}")

        # Basic validation
        if X_train.shape[0] == 0:
             raise ValueError("Loaded training data is empty.")
        if Y_train is None:
             raise ValueError("Labels (Y_train) could not be loaded for training.")
        if Y_train is not None and X_train.shape[0] != Y_train.shape[0]:
             raise ValueError(f"Number of samples mismatch between features ({X_train.shape[0]}) and labels ({Y_train.shape[0]}).")


        # Ensure data types are float32 for training
        if X_train.dtype != np.float32:
            logging.warning(f"Converting X_train dtype from {X_train.dtype} to float32.")
            X_train = X_train.astype(np.float32)
        if Y_train is not None and Y_train.dtype != np.float32:
            # Check if Y_train is already numeric before casting
            if not np.issubdtype(Y_train.dtype, np.number):
                 raise TypeError(f"Loaded labels (Y_train) have non-numeric dtype {Y_train.dtype} before casting.")
            logging.warning(f"Converting Y_train dtype from {Y_train.dtype} to float32.")
            Y_train = Y_train.astype(np.float32)


    except Exception as e:
        logging.error(f"Error loading training data from {data_path}: {e}")
        raise

    # --- Step 3: Prepare Optimizer and Compile ---
    try:
        # Get the loss and metrics from the loaded model config if available
        # Default loss/metrics if not found in config (should generally be there)
        try:
            config = model.get_config()
            # Loss might be stored directly or under 'compile_config'
            loss_func = config.get('loss') or model.loss or 'binary_crossentropy'
            metrics_list = config.get('metrics') or ['accuracy']
            # If metrics are stored under compile_config
            if not metrics_list and 'compile_config' in config and config['compile_config']:
                 metrics_list = config['compile_config'].get('metrics', ['accuracy'])
            if not loss_func and 'compile_config' in config and config['compile_config']:
                 loss_func = config['compile_config'].get('loss', 'binary_crossentropy')

        except Exception as config_e:
            logging.warning(f"Could not get loss/metrics from model config ({config_e}). Using defaults.")
            loss_func = 'binary_crossentropy'
            metrics_list = ['accuracy']

        # Create a new optimizer instance with the specified learning rate
        # Note: This approach creates a new optimizer state. If resuming training
        # with the exact previous optimizer state is crucial, loading the saved
        # optimizer and modifying its learning rate is more complex but possible.
        # Creating a new instance is simpler and often sufficient.
        # This might trigger a Keras warning about skipping variable loading if
        # the saved model had a different optimizer state.
        optimizer_name = 'adam' # Default or get from config if needed
        try:
            # Try to get the optimizer name from the loaded model if available
            if model.optimizer:
                optimizer_name = model.optimizer.__class__.__name__.lower()
            elif 'compile_config' in config and config['compile_config']:
                 opt_config = config['compile_config'].get('optimizer', {})
                 if isinstance(opt_config, dict):
                      optimizer_name = opt_config.get('class_name', 'adam').lower()
                 elif isinstance(opt_config, str):
                      optimizer_name = opt_config.lower()

            logging.info(f"Creating new '{optimizer_name}' optimizer instance with learning_rate={learning_rate}")
            # Get the optimizer class and instantiate it
            optimizer_cls = tf.keras.optimizers.get(optimizer_name) # Get class from registry
            new_optimizer = optimizer_cls(learning_rate=learning_rate)

        except Exception as opt_e:
            logging.warning(f"Could not determine original optimizer type ({opt_e}). Using new Adam optimizer.")
            new_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        logging.info(f"Compiling model with optimizer='{new_optimizer.__class__.__name__}', loss='{loss_func}', learning_rate={learning_rate}, metrics={metrics_list}")
        model.compile(
            optimizer=new_optimizer,
            loss=loss_func,
            metrics=metrics_list
        )
        logging.info("Model compiled successfully.")
        model.summary(print_fn=logging.info) # Print summary after compilation

    except Exception as e:
        logging.error(f"Error preparing optimizer or compiling model: {e}", exc_info=True)
        raise


    # --- Step 4: Train the Model ---
    try:
        logging.info("Starting model training...")
        # Model should be built if loaded correctly, Functional API models are built on instantiation
        # No explicit build step needed here typically

        history = model.fit(
            X_train,
            Y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=2 # Print progress bar per epoch
        )
        logging.info("Model training completed.")

        # Log final metrics robustly
        final_loss = history.history.get('loss', [None])[-1]
        final_accuracy = history.history.get('accuracy', [None])[-1] # Keras default metric name
        final_acc_alt = history.history.get('acc', [None])[-1] # Older Keras name

        if final_loss is not None:
             logging.info(f"Final training loss: {final_loss:.4f}")
        if final_accuracy is not None:
            logging.info(f"Final training accuracy: {final_accuracy:.4f}")
        elif final_acc_alt is not None:
             logging.info(f"Final training accuracy: {final_acc_alt:.4f}")


    except Exception as e:
        logging.error(f"Error during model training: {e}", exc_info=True)
        raise

    # --- Step 5: Save the Trained Keras Model ---
    try:
        output_dir = os.path.dirname(model_path)
        if output_dir:
             os.makedirs(output_dir, exist_ok=True)
        model.save(model_path)
        logging.info(f"Trained Keras model saved successfully to {model_path}")
    except Exception as e:
        logging.error(f"Error saving trained Keras model to {model_path}: {e}")
        raise

    # --- Step 6: Optional ONNX Export ---
    if export_onnx_path:
        logging.info(f"Exporting trained model to ONNX format: {export_onnx_path}")
        try:
            # --- Determine Input Signature ---
            # Model should be built now after training. Try inferring from model.inputs.
            input_signature = None
            try:
                # Use model.inputs which should be defined for Functional models
                if hasattr(model, 'inputs') and model.inputs:
                    input_signature = [tf.TensorSpec(shape=inp.shape, dtype=inp.dtype, name=inp.name.split(':')[0])
                                       for inp in model.inputs]
                    logging.info(f"Inferred input signatures for ONNX export from model.inputs: {input_signature}")
                else:
                     # Fallback for models where .inputs might not be standard
                     raise ValueError("Could not access model.inputs")

            except Exception as sig_e:
                 logging.warning(f"Could not automatically infer input signature from model attributes ({sig_e}). "
                                 "Attempting signature based on training data shape.")
                 if X_train is not None:
                     try:
                         inferred_shape = (None,) + X_train.shape[1:]
                         # Use the dtype from the model's input layer if possible
                         input_dtype = tf.float32 # Default
                         if hasattr(model, 'inputs') and model.inputs:
                             input_dtype = model.inputs[0].dtype
                         elif hasattr(model, 'input') and model.input: # Fallback
                              input_dtype = model.input.dtype

                         input_signature = [tf.TensorSpec(shape=inferred_shape, dtype=input_dtype)]
                         logging.info(f"Using input signature based on training data shape: {input_signature}")
                     except Exception as shape_e:
                          logging.warning(f"Could not infer signature from data shape: {shape_e}")

            if not input_signature:
                 logging.error("Failed to determine input signature for ONNX export. Skipping conversion.")
            else:
                 # --- Call Conversion ---
                 convert_to_onnx(
                     keras_model_path=model_path, # Use the just-saved trained model
                     output_onnx_path=export_onnx_path,
                     input_signature_list=input_signature # Pass the inferred signature list
                 )
                 logging.info(f"Model successfully exported to ONNX: {export_onnx_path}")
        except Exception as e:
            # Log error but don't stop the whole process if Keras saving succeeded
            logging.error(f"Error exporting model to ONNX format: {e}", exc_info=True)
