import tensorflow as tf
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mapping from string names to Keras layer classes
LAYER_MAP = {
    'dense': tf.keras.layers.Dense,
    'conv1d': tf.keras.layers.Conv1D,
    'conv2d': tf.keras.layers.Conv2D,
    'flatten': tf.keras.layers.Flatten,
    'maxpool1d': tf.keras.layers.MaxPooling1D,
    'maxpool2d': tf.keras.layers.MaxPooling2D,
    'dropout': tf.keras.layers.Dropout,
    # Add other layers as needed (e.g., LSTM, GRU, BatchNormalization)
}

def _parse_list_str(list_str, dtype=int):
    """Helper to parse comma-separated string into a list of specified type."""
    if list_str is None:
        return []
    items = []
    for item in list_str.split(','):
        item_stripped = item.strip()
        if item_stripped.lower() == 'none':
            items.append(None)
        else:
            try:
                items.append(dtype(item_stripped))
            except ValueError as e:
                raise ValueError(f"Error parsing item '{item_stripped}' in list string '{list_str}' with type {dtype}: {e}") from e
    return items

def _parse_layer_configs_flexible(layers_str):
    """
    Parses the layers string into numerical values (int/float) or None.
    Handles potential float values for dropout rates.
    """
    if not layers_str:
        return []
    configs = []
    for item in layers_str.split(','):
        item_stripped = item.strip()
        if item_stripped.lower() == 'none':
            configs.append(None)
        else:
            try:
                configs.append(int(item_stripped))
            except ValueError:
                try:
                    configs.append(float(item_stripped))
                except ValueError:
                    raise ValueError(f"Invalid layer configuration value: '{item_stripped}'. Must be integer, float, or 'None'.")
    return configs


def create_network(
    layers_str: str,
    output_path: str,
    layer_types_str: str | None = None,
    activations_str: str = "relu",
    loss: str = "binary_crossentropy",
    optimizer: str = "adam",
    model_type: str = "deterministic",
    output_distribution: str | None = None,
    kernel_sizes_str: str | None = None,
    pool_sizes_str: str | None = None,
    strides_str: str | None = None,
    padding_str: str = 'valid',
    input_shape_str: str | None = None
    ):
    """
    Create a neural network using the Keras Functional API based on specified parameters
    and save the architecture.

    Args:
        layers_str (str): Comma-separated list of neurons/filters/dropout_rate per layer.
                          Use 'None' for layers like Flatten or Pooling.
                          For Dense networks without input_shape_str, the first number is input_dim.
                          For Dropout layers, the value should be the float dropout rate (e.g., 0.3).
                          For probabilistic regression (e.g., Gaussian), the last Dense layer's unit
                          count specifies the number of target dimensions. This will be internally
                          doubled to output mean and log-variance.
        output_path (str): Path to save the created Keras model (.keras format).
        layer_types_str (str | None): Comma-separated list of layer types.
                                      Defaults to 'dense' if None or if first layer is Dense input dim.
        activations_str (str): Comma-separated activation functions for hidden/output layers.
                               Use 'linear' or 'None' as placeholders for non-activatable layers.
        loss (str): Loss function for compiling the model.
        optimizer (str): Optimizer for compiling the model.
        model_type (str): Type of model, e.g., "deterministic", "probabilistic_regression".
        output_distribution (str | None): Probability distribution for the output layer.
        kernel_sizes_str (str | None): Comma-separated kernel sizes for Conv layers.
        pool_sizes_str (str | None): Comma-separated pool sizes for MaxPooling layers.
        strides_str (str | None): Comma-separated strides for Conv/Pooling layers.
        padding_str (str): Padding type ('valid' or 'same'). Can be comma-separated.
        input_shape_str (str | None): Explicit input shape (required for CNNs, e.g., 'ts,feat' or 'h,w,c').
                                      This should NOT include the batch dimension.

    Raises:
        ValueError: If configurations are inconsistent or unsupported.
        Exception: For errors during model building or saving.
    """
    logging.info(f"Creating network (Functional API). Model Type: {model_type}, Output Distribution: {output_distribution}")
    logging.info(f"Output Path: {output_path}")
    logging.info(f"Layers Config Str: {layers_str}, Layer Types Str: {layer_types_str}")
    logging.info(f"Activations Str: {activations_str}, Input Shape Str: {input_shape_str}")

    layer_configs_raw = _parse_layer_configs_flexible(layers_str)

    # Determine the actual input shape for the tf.keras.Input layer
    # This shape should NOT include the batch dimension.
    final_input_shape_tuple = None
    layer_configs_for_processing = list(layer_configs_raw) # Make a mutable copy

    if input_shape_str:
        final_input_shape_tuple = tuple(_parse_list_str(input_shape_str, int))
        logging.info(f"Using explicit input_shape_str for tf.keras.Input: {final_input_shape_tuple}")
    elif layer_configs_raw and isinstance(layer_configs_raw[0], int) and \
         (layer_types_str is None or layer_types_str.split(',')[0].strip().lower() == 'dense'):
        # First element of layers_str is input_dim for a Dense network
        input_dim_for_dense = layer_configs_for_processing.pop(0)
        final_input_shape_tuple = (input_dim_for_dense,)
        logging.info(f"Interpreted first value from layers_str ({input_dim_for_dense}) as input_dim for tf.keras.Input: {final_input_shape_tuple}")
    else:
        raise ValueError("Input shape/dimension could not be determined. "
                         "Provide --input-shape, or ensure the first layer is Dense with its input dimension specified first in --layers.")

    if not final_input_shape_tuple: # Should be caught by the logic above
        raise ValueError("Critical error: final_input_shape_tuple not set.")

    if layer_types_str:
        layer_types = [lt.strip().lower() for lt in layer_types_str.split(',')]
    else: # Default to all dense layers if not specified
        layer_types = ['dense'] * len(layer_configs_for_processing)

    activations = [act.strip().lower() if act.strip().lower() not in ['none', ''] else 'linear' for act in activations_str.split(',')]
    kernel_sizes = _parse_list_str(kernel_sizes_str, int)
    pool_sizes = _parse_list_str(pool_sizes_str, int)
    strides = _parse_list_str(strides_str, int)
    paddings = [p.strip().lower() for p in padding_str.split(',')]

    num_layers_to_build = len(layer_configs_for_processing)
    if len(layer_types) != num_layers_to_build:
        raise ValueError(f"Mismatch: {num_layers_to_build} layer configurations derived from --layers (after input_dim handling), "
                         f"but {len(layer_types)} layer types specified ('{layer_types_str}').")

    # Activation broadcasting logic (simplified for Functional API as activations are per layer)
    if len(activations) == 1 and num_layers_to_build > 1:
        single_activation = activations[0]
        activations = [single_activation] * num_layers_to_build # Apply to all, let layer decide if it uses it
        logging.info(f"Applied single activation '{single_activation}' to all {num_layers_to_build} configured layers.")
    elif len(activations) != num_layers_to_build:
        raise ValueError(
            f"Mismatch: Provided {len(activations)} activations ('{activations_str}'). "
            f"Expected {num_layers_to_build} activations, one for each layer specified in --layers (after input_dim)."
        )

    if len(paddings) == 1: paddings = paddings * num_layers_to_build


    is_probabilistic_gaussian_regression = (
        model_type == 'probabilistic_regression' and
        output_distribution == 'gaussian' and
        layer_types and layer_types[-1] == 'dense' and
        isinstance(layer_configs_for_processing[-1], int)
    )

    if is_probabilistic_gaussian_regression:
        num_target_dimensions = layer_configs_for_processing[-1]
        if num_target_dimensions <= 0:
            raise ValueError("For Gaussian probabilistic regression, the last layer unit count must be positive.")
        layer_configs_for_processing[-1] = num_target_dimensions * 2
        logging.info(f"Adjusted final Dense layer units to {layer_configs_for_processing[-1]} for Gaussian regression.")

    # --- Build the Model using Functional API ---
    inputs = tf.keras.Input(shape=final_input_shape_tuple, name="input_layer") # Explicit Input layer
    x = inputs # Start with the input tensor

    kernel_idx, pool_idx, stride_idx, padding_idx = 0, 0, 0, 0

    for i, layer_type in enumerate(layer_types):
        layer_config_args = {}
        current_config_value = layer_configs_for_processing[i]
        layer_activation = activations[i] # Each layer gets its specified activation

        logging.debug(f"Preparing Layer {i}: Type={layer_type}, ConfigVal={current_config_value}, Activation={layer_activation}")

        if layer_type not in LAYER_MAP:
            raise ValueError(f"Unsupported layer type: '{layer_type}'. Supported: {list(LAYER_MAP.keys())}")
        LayerClass = LAYER_MAP[layer_type]

        current_padding = paddings[padding_idx % len(paddings)]

        if layer_type == 'dense':
            if not isinstance(current_config_value, int):
                 raise ValueError(f"Config for Dense layer {i} must be int (units), got: {current_config_value}")
            layer_config_args['units'] = current_config_value
            if layer_activation not in ['linear', 'none']:
                layer_config_args['activation'] = layer_activation
        elif layer_type.startswith('conv'):
            if not isinstance(current_config_value, int):
                 raise ValueError(f"Config for {layer_type} layer {i} must be int (filters), got: {current_config_value}")
            layer_config_args['filters'] = current_config_value
            if kernel_idx < len(kernel_sizes):
                layer_config_args['kernel_size'] = kernel_sizes[kernel_idx]; kernel_idx += 1
            else: raise ValueError(f"Missing kernel size for {layer_type} layer {i}")
            if stride_idx < len(strides): layer_config_args['strides'] = strides[stride_idx]
            layer_config_args['padding'] = current_padding
            if layer_activation not in ['linear', 'none']:
                layer_config_args['activation'] = layer_activation
            stride_idx += 1; padding_idx += 1
        elif layer_type.startswith('maxpool'):
            if current_config_value is not None: logging.warning(f"Ignoring config '{current_config_value}' for {layer_type} at index {i}.")
            if pool_idx < len(pool_sizes):
                layer_config_args['pool_size'] = pool_sizes[pool_idx]; pool_idx += 1
            else: raise ValueError(f"Missing pool size for {layer_type} layer {i}")
            if stride_idx < len(strides): layer_config_args['strides'] = strides[stride_idx]
            layer_config_args['padding'] = current_padding
            stride_idx += 1; padding_idx += 1
        elif layer_type == 'flatten':
            if current_config_value is not None: logging.warning(f"Ignoring config '{current_config_value}' for Flatten at index {i}.")
        elif layer_type == 'dropout':
             if not isinstance(current_config_value, float) or not (0 <= current_config_value < 1): # Allow 0 for dropout
                  raise ValueError(f"Config for Dropout at index {i} must be float rate [0,1), got: {current_config_value}")
             layer_config_args['rate'] = current_config_value

        # Add the layer functionally
        logging.debug(f"Adding Layer instance: {LayerClass.__name__} with config: {layer_config_args}")
        try:
            current_layer_instance = LayerClass(**layer_config_args)
            x = current_layer_instance(x) # Apply layer to the tensor x
        except Exception as add_layer_e:
            logging.error(f"Failed to instantiate or call layer {i} ({LayerClass.__name__}) with args {layer_config_args}: {add_layer_e}", exc_info=True)
            raise

    # Create the Keras Model
    model = tf.keras.Model(inputs=inputs, outputs=x, name=f"{model_type}_functional_model")

    try:
        logging.info(f"Compiling model with optimizer='{optimizer}', loss='{loss}'")
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    except Exception as e:
        logging.error(f"Error compiling model: {e}")
        raise

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        model.save(output_path)
        logging.info(f"Model successfully created using Functional API and saved to {output_path}")
        model.summary(print_fn=logging.info)
    except Exception as e:
        logging.error(f"Error saving model to {output_path}: {e}")
        raise
