# sygnals_nn/convert.py
# -*- coding: utf-8 -*-
import tensorflow as tf
import tf2onnx # Or import keras_onnx if using that library
import os
import logging
import numpy as np # Import numpy

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
    If the loaded model is Sequential, it attempts to convert it to a
    Functional model before ONNX conversion to avoid potential issues.

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
        # Load the full model. Keep compile=True for safety.
        model = tf.keras.models.load_model(keras_model_path, compile=True)
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
            logging.warning(f"Evaluating input signature string: {input_signature_str}")
            # Restrict eval context for security
            eval_context = {'tf': tf, 'None': None, 'list': list, 'tuple': tuple}
            input_signature = eval(input_signature_str, {"__builtins__": {}}, eval_context) # Safer eval
            if not isinstance(input_signature, list):
                 input_signature = [input_signature]
            if not all(isinstance(spec, tf.TensorSpec) for spec in input_signature):
                 raise TypeError("Evaluated input signature must be a list of tf.TensorSpec objects.")
            logging.info(f"Using evaluated input signature: {input_signature}")
        except Exception as e:
            logging.error(f"Error evaluating input_signature_str: {e}. Please provide a valid Python list of tf.TensorSpec objects.")
            raise ValueError("Invalid input_signature_str format.") from e
    else:
        # Try to infer signature from the loaded model
        try:
            logging.info("Attempting to infer input signature from model...")
            # Check if the model has inputs defined (common after build/train/Functional API)
            if hasattr(model, 'inputs') and model.inputs:
                 input_signature = [tf.TensorSpec(shape=inp.shape, dtype=inp.dtype, name=inp.name.split(':')[0])
                                    for inp in model.inputs]
                 logging.info(f"Inferred input signatures from model.inputs: {input_signature}")
            # Fallback for Sequential models that might have _build_input_shape after build/fit
            elif hasattr(model, '_build_input_shape') and model._build_input_shape is not None:
                 build_shape = model._build_input_shape
                 dtype = tf.float32 # Assume float32 dtype if not otherwise known
                 if model.layers: # Try getting dtype from first layer
                      try:
                           layer_dtype = getattr(model.layers[0], 'dtype', tf.float32)
                           dtype = tf.dtypes.as_dtype(layer_dtype) # Convert string like 'float32'
                      except Exception: pass # Stick with default float32
                 signature_shape = (None,) + tuple(build_shape[1:]) # Add None for batch dimension
                 input_signature = [tf.TensorSpec(shape=signature_shape, dtype=dtype)]
                 logging.info(f"Inferred single input signature from _build_input_shape: {input_signature}")
            else:
                 # If no inputs and no _build_input_shape, try getting shape from config
                 try:
                     config = model.get_config()
                     first_layer_config = config['layers'][0]['config']
                     if 'batch_input_shape' in first_layer_config:
                         sig_shape = tuple(first_layer_config['batch_input_shape'])
                         sig_dtype = tf.dtypes.as_dtype(first_layer_config.get('dtype', 'float32'))
                         input_signature = [tf.TensorSpec(shape=sig_shape, dtype=sig_dtype)]
                         logging.info(f"Inferred input signature from model config's batch_input_shape: {input_signature}")
                     else:
                         raise ValueError("Model config lacks 'batch_input_shape' in the first layer.")
                 except Exception as config_e:
                     raise ValueError(f"Could not infer signature from model.inputs, _build_input_shape, or config: {config_e}")

        except Exception as e:
            logging.error(f"Could not automatically infer input signature: {e}. "
                           "ONNX conversion might fail. Provide --input-signature explicitly.")
            raise ValueError("Failed to determine input signature for ONNX conversion.") from e

    # --- Convert Sequential to Functional if necessary ---
    # This helps resolve issues like missing 'output_names' attribute for tf2onnx
    functional_model = None
    if isinstance(model, tf.keras.Sequential):
        logging.info("Detected Sequential model. Converting to Functional API model for ONNX export.")
        try:
            # Ensure we have a single, clear input signature to proceed
            if not input_signature or len(input_signature) != 1:
                 raise ValueError("Cannot automatically convert Sequential to Functional without a single, clear input signature. Provide --input-signature if needed.")
            input_spec = input_signature[0]

            # Create an Input layer using the signature (shape excludes batch dim)
            input_shape_no_batch = input_spec.shape[1:]
            functional_input = tf.keras.Input(shape=input_shape_no_batch, dtype=input_spec.dtype, name=input_spec.name or 'input_layer')

            # Rebuild the model functionally layer by layer
            functional_output = functional_input
            for layer in model.layers:
                functional_output = layer(functional_output)

            # Create the final Functional model
            functional_model = tf.keras.Model(inputs=functional_input, outputs=functional_output, name=model.name + "_functional")

            # Re-compile the functional model with the original optimizer/loss/metrics
            # Get optimizer config if possible, handle potential issues
            try:
                current_optimizer = model.optimizer
                if isinstance(current_optimizer, str):
                     logging.warning(f"Original optimizer was string '{current_optimizer}'. Using new Adam.")
                     new_optimizer = tf.keras.optimizers.Adam() # Use default LR for recompile
                elif hasattr(current_optimizer, 'get_config'):
                     optimizer_config = current_optimizer.get_config()
                     new_optimizer = current_optimizer.__class__.from_config(optimizer_config)
                else:
                     logging.warning("Could not get original optimizer config. Using new Adam.")
                     new_optimizer = tf.keras.optimizers.Adam()
            except Exception as opt_e:
                 logging.warning(f"Error getting optimizer config ({opt_e}). Using new Adam.")
                 new_optimizer = tf.keras.optimizers.Adam()

            functional_model.compile(optimizer=new_optimizer, loss=model.loss, metrics=model.metrics_names[1:]) # Assumes first metric is loss
            logging.info("Successfully converted Sequential to Functional model and recompiled.")
            functional_model.summary(print_fn=logging.info)
            model_to_convert = functional_model # Use the new functional model for conversion
        except Exception as func_e:
            logging.warning(f"Failed to convert Sequential to Functional model: {func_e}. Proceeding with original Sequential model, conversion might fail.")
            model_to_convert = model # Fallback to original model
    else:
        # Model is already Functional or subclassed, use it directly
        logging.info("Model is not Sequential, proceeding with original model for conversion.")
        model_to_convert = model


    # --- Build model explicitly (Optional but potentially helpful) ---
    # This step might be less critical if the Functional conversion succeeded,
    # but can be kept as a fallback or for non-Sequential models.
    # We'll skip the explicit build here to rely on the Functional conversion or the loaded state.
    # logging.info("Skipping explicit build step after potential Functional conversion.")


    # --- Perform ONNX Conversion ---
    try:
        logging.info(f"Converting model to ONNX (opset {opset})...")
        # Ensure output directory exists
        output_dir = os.path.dirname(output_onnx_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Use the potentially converted functional model
        model_proto, external_tensor_storage = tf2onnx.convert.from_keras(
            model=model_to_convert, # Use the functional model if created
            input_signature=input_signature, # Pass the list of tf.TensorSpec objects
            opset=opset,
            output_path=output_onnx_path # tf2onnx can save directly
        )
        logging.info(f"Model successfully converted and saved to ONNX: {output_onnx_path}")

    except ImportError:
        logging.error("`tf2onnx` library is required for ONNX conversion. Please install it (`pip install tf2onnx`).")
        raise
    except Exception as e:
        logging.error(f"Error during ONNX conversion: {e}")
        # Provide more context if possible
        logging.error(f"Conversion attempted with model type: {type(model_to_convert)}")
        logging.error(f"Model Input Signature used: {input_signature}")
        logging.error("Ensure the input signature matches the model's expected input.")
        logging.error("Check compatibility between TensorFlow, Keras, and tf2onnx versions.")
        # Check for the specific error seen in tests
        if "'Sequential' object has no attribute 'output_names'" in str(e):
             logging.error("This specific error often occurs if the model wasn't properly built or if Sequential to Functional conversion failed.")
        raise
