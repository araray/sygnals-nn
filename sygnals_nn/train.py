import tensorflow as tf
import os
import logging
from sygnals_nn.utils import load_data
# Import ONNX conversion function if exporting after training
from sygnals_nn.convert import convert_to_onnx

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
        model = tf.keras.models.load_model(model_path)
        logging.info("Model loaded successfully.")
        model.summary(print_fn=logging.info)
    except Exception as e:
        logging.error(f"Error loading Keras model from {model_path}: {e}")
        raise

    # --- Step 2: Load the Training Data ---
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
        logging.info(f"Training data loaded. X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")

        # Basic validation
        if X_train.shape[0] != Y_train.shape[0]:
             raise ValueError(f"Number of samples mismatch between features ({X_train.shape[0]}) and labels ({Y_train.shape[0]}).")
        if X_train.shape[0] == 0:
             raise ValueError("Loaded training data is empty.")

    except Exception as e:
        logging.error(f"Error loading training data from {data_path}: {e}")
        raise

    # --- Step 3: Recompile the Model with Specified Learning Rate ---
    # It's generally good practice to recompile to ensure the optimizer and
    # learning rate from the command line are used, overriding any saved state.
    try:
        logging.info(f"Recompiling model with optimizer: {model.optimizer.name}, loss: {model.loss}, learning_rate: {learning_rate}")
        # Get the existing optimizer config and update the learning rate
        optimizer_config = model.optimizer.get_config()
        optimizer_config['learning_rate'] = learning_rate
        new_optimizer = model.optimizer.__class__.from_config(optimizer_config)

        # Recompile with the new optimizer instance and existing loss/metrics
        model.compile(
            optimizer=new_optimizer,
            loss=model.loss, # Use the loss function the model was originally compiled with
            metrics=model.metrics_names[1:] # Keep existing metrics (excluding the loss itself)
        )
        logging.info("Model recompiled successfully.")
    except Exception as e:
        logging.error(f"Error recompiling model: {e}")
        # Decide if you want to proceed with the old compilation or raise error
        raise # Safer to raise error if recompilation fails


    # --- Step 4: Train the Model ---
    try:
        logging.info("Starting model training...")
        history = model.fit(
            X_train,
            Y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=2 # Show progress per epoch
            # Add validation_split or validation_data later if needed
            # validation_split=0.2
        )
        logging.info("Model training completed.")
        # Log final metrics
        final_loss = history.history['loss'][-1]
        final_metrics = {m: history.history[m][-1] for m in model.metrics_names[1:]} # Get last value for metrics
        logging.info(f"Final training loss: {final_loss:.4f}")
        for name, value in final_metrics.items():
            logging.info(f"Final training {name}: {value:.4f}")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

    # --- Step 5: Save the Trained Keras Model ---
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path) # Overwrite the original file with the trained version
        logging.info(f"Trained Keras model saved successfully to {model_path}")
    except Exception as e:
        logging.error(f"Error saving trained Keras model to {model_path}: {e}")
        # Decide if you want to raise error or just warn
        raise

    # --- Step 6: Optional ONNX Export ---
    if export_onnx_path:
        logging.info(f"Exporting trained model to ONNX format: {export_onnx_path}")
        try:
            # Determine input signature from the model if not provided explicitly
            # This is crucial for ONNX conversion.
            # We infer it from the first layer's input specification.
            input_signature = None
            try:
                # tf.keras.layers.InputLayer is often the first layer implicitly
                input_spec = model.inputs[0].spec
                input_signature = [input_spec]
                logging.info(f"Inferred input signature for ONNX export: {input_signature}")
            except Exception as sig_e:
                logging.warning(f"Could not automatically infer input signature for ONNX export: {sig_e}. "
                                "Conversion might fail or require manual --input-signature in 'convert' command.")
                # Fallback: try using the shape of the training data batch
                if X_train is not None:
                    try:
                         # Use None for batch size dimension
                        inferred_shape = (None,) + X_train.shape[1:]
                        input_signature = [tf.TensorSpec(shape=inferred_shape, dtype=tf.float32)]
                        logging.info(f"Using input signature based on training data shape: {input_signature}")
                    except Exception as shape_e:
                         logging.warning(f"Could not infer signature from data shape: {shape_e}")


            convert_to_onnx(
                keras_model_path=model_path, # Use the just-saved trained model
                output_onnx_path=export_onnx_path,
                input_signature_list=input_signature # Pass the inferred signature list
            )
            logging.info(f"Model successfully exported to ONNX: {export_onnx_path}")
        except Exception as e:
            logging.error(f"Error exporting model to ONNX format: {e}")
            # Log error but don't stop the whole process if Keras saving succeeded
            # Consider adding a flag to make ONNX export failure critical if needed
