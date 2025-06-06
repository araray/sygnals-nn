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

# --- MODIFICATION: Allow parsing floats/ints/None initially ---
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
                # Try parsing as int first
                configs.append(int(item_stripped))
            except ValueError:
                try:
                    # If int fails, try parsing as float (for dropout rates)
                    configs.append(float(item_stripped))
                except ValueError:
                    # If both fail, raise an error
                    raise ValueError(f"Invalid layer configuration value: '{item_stripped}'. Must be integer, float, or 'None'.")
    return configs
# --- END MODIFICATION ---


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
        layers_str: Comma-separated list of input_dim/neurons/filters/dropout_rate per layer.
                    Use 'None' for layers like Flatten or Pooling.
                    For Dense networks without input_shape_str, the first number is input_dim.
                    For Dropout layers, the value should be the float dropout rate (e.g., 0.3).
        output_path: Path to save the created Keras model (.keras format).
        layer_types_str: Comma-separated list of layer types (e.g., 'dense', 'conv1d', 'dropout').
                         Defaults to 'dense' if None or if first layer is Dense input dim.
        activations_str: Comma-separated activation functions for hidden/output layers.
                         Use 'linear' or 'None' as placeholders for non-activatable layers like Dropout/Flatten/Pooling.
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
    # --- MODIFICATION: Use flexible parser ---
    layer_configs_raw = _parse_layer_configs_flexible(layers_str) # e.g., [384, 128, 0.3, 64, 0.3, 96]
    # --- END MODIFICATION ---
    input_shape = tuple(_parse_list_str(input_shape_str, int)) if input_shape_str else None

    # Determine if the first layer config is input dim for Dense
    is_dense_input_dim_specified = False
    layer_configs = layer_configs_raw # Start with the raw parsed list
    if not input_shape and layer_configs and isinstance(layer_configs[0], int): # Check if first is int
         # If no explicit input_shape, assume first *integer* is input dim for Dense
         if layer_types_str is None or layer_types_str.split(',')[0].strip().lower() == 'dense':
              is_dense_input_dim_specified = True
              input_dim = layer_configs[0]
              layer_configs = layer_configs[1:] # Remove input dim from layer processing list
              logging.info(f"Interpreted first integer value ({input_dim}) as input dimension for Dense network.")
         else:
              # First layer is not Dense, but no input_shape provided
              raise ValueError("Must specify --input-shape for models not starting with a Dense layer or input dimension.")

    # Parse layer types, defaulting to dense if needed
    if layer_types_str:
        layer_types = [lt.strip().lower() for lt in layer_types_str.split(',')]
    else:
        # If layer_types not given, assume all remaining configs are for Dense layers
        if any(not isinstance(lc, int) for lc in layer_configs if lc is not None):
             raise ValueError("If --layer-types is not specified, all values in --layers (after input dim) must be integers (for Dense layers).")
        layer_types = ['dense'] * len(layer_configs)

    activations = [act.strip().lower() if act else 'linear' for act in activations_str.split(',')] # Treat empty/None as linear
    kernel_sizes = _parse_list_str(kernel_sizes_str, int)
    pool_sizes = _parse_list_str(pool_sizes_str, int)
    strides = _parse_list_str(strides_str, int)
    paddings = [p.strip().lower() for p in padding_str.split(',')]

    # --- Validate Arguments ---
    num_layers_to_build = len(layer_configs)
    if len(layer_types) != num_layers_to_build:
        raise ValueError(f"Mismatch: {num_layers_to_build} layer configurations derived from --layers ('{layers_str}' after input dim handling), "
                         f"but {len(layer_types)} layer types specified ('{layer_types_str}').")

    # Activation validation: Need one activation per layer *including* placeholders for non-activated layers
    # The number of activations must now match the number of layers being built.
    if len(activations) != num_layers_to_build:
         # Allow single activation to be broadcast *only if* all layers are activatable (e.g., all Dense)
         all_activatable = all(lt in ['dense', 'conv1d', 'conv2d'] for lt in layer_types) # Adjust if more activatable layers added
         if len(activations) == 1 and all_activatable:
              single_activation = activations[0]
              activations = [single_activation] * num_layers_to_build
              logging.info(f"Applying single activation '{single_activation}' to all {num_layers_to_build} layers.")
         else:
              raise ValueError(f"Mismatch: Provided {len(activations)} activations ('{activations_str}'), "
                               f"but expected {num_layers_to_build} (one per layer, use 'linear' or 'None' for non-activatable layers like Dropout/Flatten/Pooling).")

    # Expand single padding if needed
    if len(paddings) == 1:
        paddings = paddings * num_layers_to_build


    # --- Build the Model ---
    model = tf.keras.Sequential()
    kernel_idx, pool_idx, stride_idx, padding_idx = 0, 0, 0, 0

    for i, layer_type in enumerate(layer_types):
        layer_config_args = {}
        # Get the config value (int, float, or None) for the current layer
        current_config_value = layer_configs[i]
        current_activation = activations[i] # Get activation for this specific layer index

        logging.debug(f"Building Layer {i}: Type={layer_type}, Config={current_config_value}, Activation={current_activation}")

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

        # Configure Layer Arguments based on type
        current_padding = paddings[padding_idx % len(paddings)] # Cycle through paddings

        if layer_type == 'dense':
            if not isinstance(current_config_value, int):
                 raise ValueError(f"Configuration for Dense layer at index {i} must be an integer (units), got: {current_config_value}")
            layer_config_args['units'] = current_config_value
            if current_activation not in ['linear', 'none']:
                layer_config_args['activation'] = current_activation
        elif layer_type.startswith('conv'):
            if not isinstance(current_config_value, int):
                 raise ValueError(f"Configuration for {layer_type} layer at index {i} must be an integer (filters), got: {current_config_value}")
            layer_config_args['filters'] = current_config_value
            if kernel_idx < len(kernel_sizes):
                layer_config_args['kernel_size'] = kernel_sizes[kernel_idx]
                kernel_idx += 1
            else: raise ValueError(f"Missing kernel size for {layer_type} layer at index {i}")
            if stride_idx < len(strides):
                layer_config_args['strides'] = strides[stride_idx]
            layer_config_args['padding'] = current_padding
            if current_activation not in ['linear', 'none']:
                layer_config_args['activation'] = current_activation
            # Increment stride/padding index if used by conv
            stride_idx += 1
            padding_idx += 1
        elif layer_type.startswith('maxpool'):
            if current_config_value is not None:
                logging.warning(f"Ignoring configuration value '{current_config_value}' for {layer_type} layer at index {i}. Use 'None'.")
            if pool_idx < len(pool_sizes):
                layer_config_args['pool_size'] = pool_sizes[pool_idx]
                pool_idx += 1
            else: raise ValueError(f"Missing pool size for {layer_type} layer at index {i}")
            if stride_idx < len(strides):
                layer_config_args['strides'] = strides[stride_idx]
            layer_config_args['padding'] = current_padding
            if current_activation not in ['linear', 'none']:
                 logging.warning(f"Activation '{current_activation}' ignored for {layer_type} layer at index {i}.")
            # Increment stride/padding index if used by pool
            stride_idx += 1
            padding_idx += 1
        elif layer_type == 'flatten':
            if current_config_value is not None:
                logging.warning(f"Ignoring configuration value '{current_config_value}' for Flatten layer at index {i}. Use 'None'.")
            if current_activation not in ['linear', 'none']:
                 logging.warning(f"Activation '{current_activation}' ignored for Flatten layer at index {i}.")
            pass # No arguments needed typically
        elif layer_type == 'dropout':
             if not isinstance(current_config_value, float) or not (0 < current_config_value < 1):
                  raise ValueError(f"Configuration for Dropout layer at index {i} must be a float rate between 0 and 1 (exclusive), got: {current_config_value}")
             layer_config_args['rate'] = current_config_value
             if current_activation not in ['linear', 'none']:
                 logging.warning(f"Activation '{current_activation}' ignored for Dropout layer at index {i}.")
        else:
            # Handle other potential layers if added to LAYER_MAP
            pass


        # Add layer to model
        logging.debug(f"Adding Layer: {LayerClass.__name__} with config: {layer_config_args}")
        try:
            model.add(LayerClass(**layer_config_args))
        except Exception as add_layer_e:
            logging.error(f"Failed to add layer {i} ({LayerClass.__name__}) with args {layer_config_args}: {add_layer_e}", exc_info=True)
            raise


    # --- Compile the Model ---
    try:
        logging.info(f"Compiling model with optimizer='{optimizer}', loss='{loss}'")
        # Add accuracy as a default metric, can be overridden by user if needed via advanced options
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
