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
            items.append(None) # Keep None as None
        else:
            try:
                items.append(dtype(item_stripped))
            except ValueError as e:
                raise ValueError(f"Error parsing item '{item_stripped}' in list string '{list_str}' with type {dtype}: {e}") from e
    return items

def _parse_layer_configs(layers_str):
    """Parses the layers string into numerical values or None."""
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
                raise ValueError(f"Invalid layer configuration value: '{item_stripped}'. Must be integer or 'None'.")
    return configs


def create_network(
    layers_str: str,
    output_path: str,
    layer_types_str: str | None = None,
    activations_str: str = "relu",
    loss: str = "binary_crossentropy",
    optimizer: str = "adam",
    kernel_sizes_str: str | None = None,
    pool_sizes_str: str | None = None,
    strides_str: str | None = None,
    padding_str: str = 'valid',
    input_shape_str: str | None = None
    ):
    """
    Create a neural network based on specified parameters and save the architecture.

    Args:
        layers_str: Comma-separated list of input_dim/neurons/filters per layer.
                    Use 'None' for layers like Flatten or Pooling where units/filters aren't applicable.
                    For Dense networks without input_shape_str, the first number is input_dim.
        output_path: Path to save the created Keras model (.keras format).
        layer_types_str: Comma-separated list of layer types (e.g., 'dense', 'conv1d').
                         Defaults to 'dense' if None or if first layer is Dense input dim.
        activations_str: Comma-separated activation functions for hidden/output layers.
        loss: Loss function for compiling the model.
        optimizer: Optimizer for compiling the model.
        kernel_sizes_str: Comma-separated kernel sizes for Conv layers.
        pool_sizes_str: Comma-separated pool sizes for MaxPooling layers.
        strides_str: Comma-separated strides for Conv/Pooling layers.
        padding_str: Padding type ('valid' or 'same'). Can be comma-separated.
        input_shape_str: Explicit input shape (required for CNNs, e.g., 'ts,feat' or 'h,w,c').
    """
    logging.info(f"Creating network. Output: {output_path}")
    logging.info(f"Layers Config Str: {layers_str}")
    logging.info(f"Layer Types Str: {layer_types_str}")
    logging.info(f"Activations Str: {activations_str}")
    logging.info(f"Input Shape Str: {input_shape_str}")

    # --- Parse Arguments ---
    layer_configs = _parse_layer_configs(layers_str) # e.g., [4, 8, 1] or [None, 32, None, 10]
    input_shape = tuple(_parse_list_str(input_shape_str, int)) if input_shape_str else None

    # Determine if the first layer config is input dim for Dense
    is_dense_input_dim_specified = False
    if not input_shape and layer_configs and layer_configs[0] is not None:
         # If no explicit input_shape, assume first number is input dim for Dense
         if layer_types_str is None or layer_types_str.split(',')[0].strip().lower() == 'dense':
              is_dense_input_dim_specified = True
              input_dim = layer_configs[0]
              layer_configs = layer_configs[1:] # Remove input dim from layer processing list
              logging.info(f"Interpreted first value ({input_dim}) as input dimension for Dense network.")
         else:
              # First layer is not Dense, but no input_shape provided
              raise ValueError("Must specify --input-shape for models not starting with a Dense layer or input dimension.")


    # Parse layer types, defaulting to dense if needed
    if layer_types_str:
        layer_types = [lt.strip().lower() for lt in layer_types_str.split(',')]
    else:
        layer_types = ['dense'] * len(layer_configs) # Default to dense for remaining configs

    activations = [act.strip().lower() for act in activations_str.split(',')]
    kernel_sizes = _parse_list_str(kernel_sizes_str, int)
    pool_sizes = _parse_list_str(pool_sizes_str, int)
    strides = _parse_list_str(strides_str, int)
    paddings = [p.strip().lower() for p in padding_str.split(',')]

    # --- Validate Arguments ---
    num_layers_to_build = len(layer_configs)
    if len(layer_types) != num_layers_to_build:
        raise ValueError(f"Mismatch: {num_layers_to_build} layer configurations derived from --layers ('{layers_str}' after input dim handling), "
                         f"but {len(layer_types)} layer types specified ('{layer_types_str}').")

    # Activation validation: Need one activation per layer *excluding* input_dim and non-activated layers (Flatten, Pool)
    num_activatable_layers = sum(1 for lt in layer_types if lt not in ['flatten', 'maxpool1d', 'maxpool2d', 'dropout']) # Approx count
    if len(activations) != num_activatable_layers:
         # Allow single activation to be broadcast
         if len(activations) == 1:
              single_activation = activations[0]
              activations = [single_activation] * num_activatable_layers
              logging.info(f"Applying single activation '{single_activation}' to all {num_activatable_layers} activatable layers.")
         else:
              raise ValueError(f"Mismatch: Provided {len(activations)} activations ('{activations_str}'), "
                               f"but expected {num_activatable_layers} for the specified layer types.")

    # Expand single padding if needed (simpler logic)
    if len(paddings) == 1:
        paddings = paddings * num_layers_to_build


    # --- Build the Model ---
    model = tf.keras.Sequential()
    activation_idx = 0
    kernel_idx, pool_idx, stride_idx, padding_idx = 0, 0, 0, 0

    for i, layer_type in enumerate(layer_types):
        layer_config_args = {}
        units_or_filters = layer_configs[i]
        needs_activation = layer_type in ['dense', 'conv1d', 'conv2d'] # Add other activatable layers here

        logging.debug(f"Building Layer {i}: Type={layer_type}, Config={units_or_filters}")

        # Handle input layer definition
        if i == 0:
            if is_dense_input_dim_specified:
                # Input dim was handled, this is the first *hidden* layer
                layer_config_args['input_shape'] = (input_dim,)
            elif input_shape:
                # Explicit input shape provided (likely for CNN/RNN)
                layer_config_args['input_shape'] = input_shape
            else:
                 # Should not happen due to earlier checks, but safeguard
                 raise ValueError("Input shape/dimension could not be determined for the first layer.")

        # Select Layer Class
        if layer_type not in LAYER_MAP:
            raise ValueError(f"Unsupported layer type: '{layer_type}'. Supported types: {list(LAYER_MAP.keys())}")
        LayerClass = LAYER_MAP[layer_type]

        # Configure Layer Arguments
        current_padding = paddings[padding_idx % len(paddings)] # Cycle through paddings if list is short

        if layer_type == 'dense':
            if units_or_filters is None: raise ValueError(f"Units must be specified for Dense layer at index {i}")
            layer_config_args['units'] = units_or_filters
        elif layer_type.startswith('conv'):
            if units_or_filters is None: raise ValueError(f"Filters must be specified for {layer_type} layer at index {i}")
            layer_config_args['filters'] = units_or_filters
            if kernel_idx < len(kernel_sizes):
                layer_config_args['kernel_size'] = kernel_sizes[kernel_idx]
                kernel_idx += 1
            else: raise ValueError(f"Missing kernel size for {layer_type} layer at index {i}")
            if stride_idx < len(strides):
                layer_config_args['strides'] = strides[stride_idx]
            layer_config_args['padding'] = current_padding
        elif layer_type.startswith('maxpool'):
            if pool_idx < len(pool_sizes):
                layer_config_args['pool_size'] = pool_sizes[pool_idx]
                pool_idx += 1
            else: raise ValueError(f"Missing pool size for {layer_type} layer at index {i}")
            if stride_idx < len(strides):
                layer_config_args['strides'] = strides[stride_idx]
            layer_config_args['padding'] = current_padding
        elif layer_type == 'flatten':
            pass # No arguments needed typically
        elif layer_type == 'dropout':
             if units_or_filters is None: raise ValueError(f"Dropout rate must be specified (as float, e.g., 0.5) for Dropout layer at index {i}")
             try:
                 rate = float(units_or_filters) # Use the 'units/filters' field for rate
                 layer_config_args['rate'] = rate
             except ValueError:
                 raise ValueError(f"Invalid dropout rate '{units_or_filters}' at index {i}. Must be a float.")

        # Assign activation if applicable
        if needs_activation:
            if activation_idx < len(activations):
                layer_config_args['activation'] = activations[activation_idx]
                activation_idx += 1
            else:
                # This case should be caught by validation above, but safeguard
                logging.warning(f"Missing activation for activatable layer {layer_type} at index {i}. Using default (linear).")

        # Increment stride/padding index if used by conv or pool
        if layer_type.startswith('conv') or layer_type.startswith('maxpool'):
             stride_idx += 1
             padding_idx += 1


        # Add layer to model
        logging.debug(f"Adding Layer: {LayerClass.__name__} with config: {layer_config_args}")
        model.add(LayerClass(**layer_config_args))

    # --- Compile the Model ---
    try:
        logging.info(f"Compiling model with optimizer='{optimizer}', loss='{loss}'")
        # Add accuracy as a default metric
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    except Exception as e:
        logging.error(f"Error compiling model: {e}")
        raise

    # --- Build the model explicitly if input shape is known ---
    # This helps prevent errors during saving or conversion if the model hasn't seen data.
    if input_shape or is_dense_input_dim_specified:
        try:
            # For dense input dim, create a dummy batch shape
            build_shape = input_shape if input_shape else (input_dim,)
            # Add batch dimension (None)
            model.build(input_shape=(None,) + build_shape)
            logging.info(f"Model built with input shape: {(None,) + build_shape}")
        except Exception as e:
            logging.warning(f"Could not explicitly build model: {e}. Saving might work, but conversion could fail.")


    # --- Save the Model ---
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir: # Only create if path includes a directory
             os.makedirs(output_dir, exist_ok=True)

        model.save(output_path) # Save in Keras native format
        logging.info(f"Model successfully created and saved to {output_path}")
        model.summary(print_fn=logging.info) # Print summary after saving
    except Exception as e:
        logging.error(f"Error saving model to {output_path}: {e}")
        raise
