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


def create_network(
    layers_str: str,
    output_path: str,
    layer_types_str: str | None = None,
    activations_str: str = "relu",
    loss: str = "binary_crossentropy",
    optimizer: str = "adam",
    model_type: str = "deterministic", # New parameter
    output_distribution: str | None = None, # New parameter
    kernel_sizes_str: str | None = None,
    pool_sizes_str: str | None = None,
    strides_str: str | None = None,
    padding_str: str = 'valid',
    input_shape_str: str | None = None
    ):
    """
    Create a neural network based on specified parameters and save the architecture.

    Args:
        layers_str (str): Comma-separated list of input_dim/neurons/filters/dropout_rate per layer.
                          Use 'None' for layers like Flatten or Pooling.
                          For Dense networks without input_shape_str, the first number is input_dim.
                          For Dropout layers, the value should be the float dropout rate (e.g., 0.3).
                          For probabilistic regression (e.g., Gaussian), the last Dense layer's unit
                          count specifies the number of target dimensions (e.g., 1 for univariate).
                          This will be internally doubled to output mean and log-variance.
        output_path (str): Path to save the created Keras model (.keras format).
        layer_types_str (str | None): Comma-separated list of layer types (e.g., 'dense', 'conv1d', 'dropout').
                                      Defaults to 'dense' if None or if first layer is Dense input dim.
        activations_str (str): Comma-separated activation functions for hidden/output layers.
                               Use 'linear' or 'None' as placeholders for non-activatable layers.
                               For probabilistic Gaussian output, the final activation applies to the mean component;
                               the log-variance component is typically linear.
        loss (str): Loss function for compiling the model. Note: for probabilistic models,
                    this loss might be a placeholder, and the actual training loss (e.g., NLL)
                    will be set in the `train` step.
        optimizer (str): Optimizer for compiling the model.
        model_type (str): Type of model, e.g., "deterministic", "probabilistic_regression".
        output_distribution (str | None): The probability distribution for the output layer,
                                          e.g., "gaussian" for probabilistic_regression.
        kernel_sizes_str (str | None): Comma-separated kernel sizes for Conv layers.
        pool_sizes_str (str | None): Comma-separated pool sizes for MaxPooling layers.
        strides_str (str | None): Comma-separated strides for Conv/Pooling layers.
        padding_str (str): Padding type ('valid' or 'same'). Can be comma-separated.
        input_shape_str (str | None): Explicit input shape (required for CNNs, e.g., 'ts,feat' or 'h,w,c').

    Raises:
        ValueError: If configurations are inconsistent or unsupported.
        Exception: For errors during model building or saving.
    """
    logging.info(f"Creating network. Model Type: {model_type}, Output Distribution: {output_distribution}")
    logging.info(f"Output Path: {output_path}")
    logging.info(f"Layers Config Str: {layers_str}, Layer Types Str: {layer_types_str}")
    logging.info(f"Activations Str: {activations_str}, Input Shape Str: {input_shape_str}")

    # --- Parse Arguments ---
    layer_configs_raw = _parse_layer_configs_flexible(layers_str)
    input_shape = tuple(_parse_list_str(input_shape_str, int)) if input_shape_str else None

    is_dense_input_dim_specified = False
    layer_configs = list(layer_configs_raw) # Make a mutable copy

    if not input_shape and layer_configs and isinstance(layer_configs[0], int):
        if layer_types_str is None or layer_types_str.split(',')[0].strip().lower() == 'dense':
            is_dense_input_dim_specified = True
            input_dim = layer_configs.pop(0) # Remove input dim from layer processing list
            logging.info(f"Interpreted first integer value ({input_dim}) as input dimension for Dense network.")
        elif model_type == "deterministic": # Only raise if not probabilistic and expecting input_shape
            raise ValueError("Must specify --input-shape for models not starting with a Dense layer or input dimension.")

    if layer_types_str:
        layer_types = [lt.strip().lower() for lt in layer_types_str.split(',')]
    else:
        if any(not isinstance(lc, int) for lc in layer_configs if lc is not None):
             raise ValueError("If --layer-types is not specified, all values in --layers (after input dim) must be integers (for Dense layers).")
        layer_types = ['dense'] * len(layer_configs)

    activations = [act.strip().lower() if act.strip().lower() not in ['none', ''] else 'linear' for act in activations_str.split(',')]
    kernel_sizes = _parse_list_str(kernel_sizes_str, int)
    pool_sizes = _parse_list_str(pool_sizes_str, int)
    strides = _parse_list_str(strides_str, int)
    paddings = [p.strip().lower() for p in padding_str.split(',')]

    # --- Validate Arguments ---
    num_layers_to_build = len(layer_configs)
    if len(layer_types) != num_layers_to_build:
        raise ValueError(f"Mismatch: {num_layers_to_build} layer configurations derived from --layers ('{layers_str}' after input dim handling), "
                         f"but {len(layer_types)} layer types specified ('{layer_types_str}').")

    if len(activations) != num_layers_to_build:
        all_activatable_types = ['dense', 'conv1d', 'conv2d'] # Types that typically take activation
        # Count activatable layers among the specified layer_types
        num_activatable_layers = sum(1 for lt in layer_types if lt in all_activatable_types)

        if len(activations) == 1 and num_activatable_layers > 0 : # Single activation can be broadcast
            single_activation = activations[0]
            # Create a full list, applying single_activation to activatable, linear to others
            full_activations = []
            act_idx = 0
            for lt in layer_types:
                if lt in all_activatable_types:
                    full_activations.append(single_activation)
                else: # For Flatten, Dropout, Pooling etc.
                    full_activations.append('linear') # Placeholder
            activations = full_activations
            logging.info(f"Applied single activation '{single_activation}' to activatable layers, 'linear' to others.")
            if len(activations) != num_layers_to_build: # Should match now
                 raise ValueError("Internal error in activation broadcasting logic.")
        elif len(activations) != num_activatable_layers and len(activations) != num_layers_to_build :
             raise ValueError(
                f"Mismatch: Provided {len(activations)} activations ('{activations_str}'). "
                f"Expected {num_activatable_layers} (for activatable layers like Dense/Conv) "
                f"or {num_layers_to_build} (if providing placeholders like 'linear' for non-activatable layers). "
                f"Activatable layers found: {num_activatable_layers}. Total layers to build: {num_layers_to_build}."
            )
        # If len(activations) == num_layers_to_build, assume user provided placeholders correctly.

    if len(paddings) == 1:
        paddings = paddings * num_layers_to_build

    # --- Adjustments for Probabilistic Models ---
    is_probabilistic_gaussian_regression = (
        model_type == 'probabilistic_regression' and
        output_distribution == 'gaussian' and
        layer_types and layer_types[-1] == 'dense' and
        isinstance(layer_configs[-1], int) # Ensure last layer config is number of target dims
    )

    if is_probabilistic_gaussian_regression:
        num_target_dimensions = layer_configs[-1]
        if num_target_dimensions <= 0:
            raise ValueError("For Gaussian probabilistic regression, the last layer unit count (target dimensions) must be positive.")
        layer_configs[-1] = num_target_dimensions * 2 # mean + log_variance for each dimension
        logging.info(f"Adjusted final Dense layer units to {layer_configs[-1]} for {num_target_dimensions} target dimension(s) in Gaussian probabilistic regression (mean + log_variance).")
        # The activation for the final layer (mean and log_var) will be handled by the loop.
        # If a single activation was provided for the output (e.g., 'linear'), it will apply to both.


    # --- Build the Model ---
    model = tf.keras.Sequential()
    kernel_idx, pool_idx, stride_idx, padding_idx = 0, 0, 0, 0
    # Track activation index for activatable layers
    current_activation_idx = 0

    for i, layer_type in enumerate(layer_types):
        layer_config_args = {}
        current_config_value = layer_configs[i]

        # Determine activation for this layer
        # If len(activations) matches num_layers_to_build, use activations[i]
        # Otherwise, use activations[current_activation_idx] for activatable layers
        layer_activation = 'linear' # Default for non-activatable or if not specified
        if layer_type in ['dense', 'conv1d', 'conv2d']:
            if current_activation_idx < len(activations):
                layer_activation = activations[current_activation_idx]
                current_activation_idx +=1
            elif len(activations) == num_layers_to_build: # if user provided full list with placeholders
                layer_activation = activations[i]

        logging.debug(f"Building Layer {i}: Type={layer_type}, Config={current_config_value}, Activation for layer: {layer_activation}")

        if i == 0: # Input layer definition
            if is_dense_input_dim_specified:
                layer_config_args['input_shape'] = (input_dim,)
            elif input_shape:
                layer_config_args['input_shape'] = input_shape
            elif not layer_configs: # Case of creating a model with only an Input layer (e.g. for a preprocessor)
                 pass # No actual layer to add, input_shape is defined by data
            else:
                 raise ValueError("Input shape/dimension could not be determined for the first layer.")

        if layer_type not in LAYER_MAP:
            raise ValueError(f"Unsupported layer type: '{layer_type}'. Supported types: {list(LAYER_MAP.keys())}")
        LayerClass = LAYER_MAP[layer_type]

        current_padding = paddings[padding_idx % len(paddings)]

        if layer_type == 'dense':
            if not isinstance(current_config_value, int):
                 raise ValueError(f"Configuration for Dense layer at index {i} must be an integer (units), got: {current_config_value}")
            layer_config_args['units'] = current_config_value
            if layer_activation not in ['linear', 'none']: # 'none' might come from user input
                layer_config_args['activation'] = layer_activation
        elif layer_type.startswith('conv'):
            if not isinstance(current_config_value, int):
                 raise ValueError(f"Configuration for {layer_type} layer at index {i} must be an integer (filters), got: {current_config_value}")
            layer_config_args['filters'] = current_config_value
            if kernel_idx < len(kernel_sizes):
                layer_config_args['kernel_size'] = kernel_sizes[kernel_idx]; kernel_idx += 1
            else: raise ValueError(f"Missing kernel size for {layer_type} layer at index {i}")
            if stride_idx < len(strides): layer_config_args['strides'] = strides[stride_idx]
            layer_config_args['padding'] = current_padding
            if layer_activation not in ['linear', 'none']:
                layer_config_args['activation'] = layer_activation
            stride_idx += 1; padding_idx += 1
        elif layer_type.startswith('maxpool'):
            if current_config_value is not None: logging.warning(f"Ignoring config '{current_config_value}' for {layer_type} at index {i}. Use 'None'.")
            if pool_idx < len(pool_sizes):
                layer_config_args['pool_size'] = pool_sizes[pool_idx]; pool_idx += 1
            else: raise ValueError(f"Missing pool size for {layer_type} layer at index {i}")
            if stride_idx < len(strides): layer_config_args['strides'] = strides[stride_idx]
            layer_config_args['padding'] = current_padding
            if layer_activation not in ['linear', 'none']: logging.warning(f"Activation '{layer_activation}' ignored for {layer_type} at index {i}.")
            stride_idx += 1; padding_idx += 1
        elif layer_type == 'flatten':
            if current_config_value is not None: logging.warning(f"Ignoring config '{current_config_value}' for Flatten at index {i}. Use 'None'.")
            if layer_activation not in ['linear', 'none']: logging.warning(f"Activation '{layer_activation}' ignored for Flatten at index {i}.")
        elif layer_type == 'dropout':
             if not isinstance(current_config_value, float) or not (0 < current_config_value < 1):
                  raise ValueError(f"Config for Dropout at index {i} must be float rate (0,1), got: {current_config_value}")
             layer_config_args['rate'] = current_config_value
             if layer_activation not in ['linear', 'none']: logging.warning(f"Activation '{layer_activation}' ignored for Dropout at index {i}.")

        logging.debug(f"Adding Layer: {LayerClass.__name__} with config: {layer_config_args}")
        try:
            model.add(LayerClass(**layer_config_args))
        except Exception as add_layer_e:
            logging.error(f"Failed to add layer {i} ({LayerClass.__name__}) with args {layer_config_args}: {add_layer_e}", exc_info=True)
            raise

    # --- Compile the Model ---
    # For probabilistic models, the loss passed here might be a placeholder.
    # The actual NLL loss will be applied during the training step.
    try:
        logging.info(f"Compiling model with optimizer='{optimizer}', loss='{loss}' (Note: loss may be overridden at training time for probabilistic models)")
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy']) # Default 'accuracy' metric
    except Exception as e:
        logging.error(f"Error compiling model: {e}")
        raise

    if input_shape or is_dense_input_dim_specified:
        try:
            build_shape_final = input_shape if input_shape else (input_dim,)
            model.build(input_shape=(None,) + build_shape_final)
            logging.info(f"Model built with input shape: {(None,) + build_shape_final}")
        except Exception as e:
            logging.warning(f"Could not explicitly build model: {e}. Saving might work, but conversion could fail.")

    # --- Save the Model ---
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        model.save(output_path)
        logging.info(f"Model successfully created and saved to {output_path}")
        model.summary(print_fn=logging.info)
    except Exception as e:
        logging.error(f"Error saving model to {output_path}: {e}")
        raise
