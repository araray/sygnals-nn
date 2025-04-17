import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _parse_list_str(list_str, dtype=int):
    """Helper to parse comma-separated strings into lists of a specific type."""
    if not list_str:
        return []
    try:
        return [dtype(x.strip()) for x in list_str.split(',')]
    except ValueError as e:
        raise ValueError(f"Error parsing comma-separated string '{list_str}' as {dtype}: {e}")

def create_network(
    layers_str: str,
    layer_types_str: str | None,
    activations_str: str,
    loss: str,
    optimizer: str,
    output_path: str,
    kernel_sizes_str: str | None = None,
    pool_sizes_str: str | None = None,
    strides_str: str | None = None,
    padding_str: str | None = 'valid',
    input_shape_str: str | None = None
    ):
    """
    Creates and saves a Keras Sequential model based on CLI arguments.

    Supports Dense layers by default. Can build CNNs (Conv1D, Conv2D, MaxPooling1D/2D, Flatten)
    by specifying layer types and relevant parameters.

    Args:
        layers_str: Comma-separated string defining neurons (for Dense) or filters (for Conv).
                    The first element is often used implicitly for input_dim if input_shape isn't given.
        layer_types_str: Comma-separated string of layer types (e.g., 'dense', 'conv1d', 'maxpool1d', 'flatten').
                         If None or empty, assumes all layers are 'dense'.
        activations_str: Comma-separated string of activation functions. If single, applied to all
                         relevant layers (Dense, Conv).
        loss: Loss function name (e.g., 'binary_crossentropy').
        optimizer: Optimizer name (e.g., 'adam').
        output_path: Path to save the Keras model (.keras).
        kernel_sizes_str: Comma-separated kernel sizes for Conv layers.
        pool_sizes_str: Comma-separated pool sizes for MaxPooling layers.
        strides_str: Comma-separated strides for Conv/Pooling layers.
        padding_str: Padding type ('valid' or 'same'), can be comma-separated.
        input_shape_str: Explicit input shape (e.g., '28,28,1' or '100,10'). Overrides implicit shape.

    Raises:
        ValueError: If configuration is inconsistent or parameters are invalid.
    """
    logging.info(f"Creating network. Output: {output_path}")

    # --- Parse Arguments ---
    layer_configs = _parse_list_str(layers_str, dtype=str) # Keep as string initially for flexibility
    activations = activations_str.split(',')
    kernel_sizes = _parse_list_str(kernel_sizes_str, dtype=int)
    pool_sizes = _parse_list_str(pool_sizes_str, dtype=int)
    strides = _parse_list_str(strides_str, dtype=int)
    paddings = padding_str.split(',') if padding_str else ['valid']

    if layer_types_str:
        layer_types = [lt.strip().lower() for lt in layer_types_str.split(',')]
        logging.info(f"Using specified layer types: {layer_types}")
    else:
        # Default to Dense layers if type not specified
        # Need N-1 types for N layer_configs (input -> layer1 -> layer2 ...)
        layer_types = ['dense'] * (len(layer_configs) -1) # Assuming first layer_config is input dim
        logging.info(f"No layer types specified, defaulting to: {layer_types}")


    num_layers_defined = len(layer_types)
    if num_layers_defined != len(layer_configs) - 1:
         raise ValueError(f"Mismatch: {len(layer_configs)} layer sizes/filters provided ({layers_str}), "
                          f"but {num_layers_defined} layer types specified ({layer_types_str}). "
                          f"Need N-1 layer types for N layer sizes (input -> layer1 -> ...).")


    # Expand single activation/padding to all layers if needed
    if len(activations) == 1:
        activations = activations * num_layers_defined
    if len(paddings) == 1:
        paddings = paddings * num_layers_defined

    if len(activations) != num_layers_defined:
        raise ValueError(f"Mismatch: {num_layers_defined} layers require activation, but {len(activations)} provided ({activations_str}).")
    # Padding check might be too strict, relax it or apply only where needed
    # if len(paddings) != num_layers_defined:
    #     raise ValueError(f"Mismatch: {num_layers_defined} layers specified, but {len(paddings)} paddings provided ({padding_str}).")


    input_shape = None
    if input_shape_str:
        input_shape = tuple(_parse_list_str(input_shape_str, dtype=int))
        logging.info(f"Using explicit input shape: {input_shape}")
    else:
        # Try to infer input dim from the first element of layers_str for simple Dense networks
        try:
            input_dim_for_dense = int(layer_configs[0])
            logging.info(f"Inferring input dimension for first Dense layer: {input_dim_for_dense}")
        except ValueError:
             input_dim_for_dense = None
             # Require input_shape if first layer isn't Dense or input can't be parsed as int
             if 'dense' in layer_types[0] : # Check if first *actual* layer is dense
                 logging.warning("Could not infer input dimension from first element of --layers. "
                                 "Specify --input-shape if the first layer is Dense.")
             elif any(lt in ['conv1d', 'conv2d'] for lt in layer_types):
                 raise ValueError("Must specify --input-shape for models starting with Conv layers.")


    # --- Build the Model ---
    model = tf.keras.Sequential(name="SygnalsNN_Model")
    kernel_idx, pool_idx, stride_idx, padding_idx = 0, 0, 0, 0

    for i, layer_type in enumerate(layer_types):
        layer_units_or_filters = int(layer_configs[i+1]) # Config for the *current* layer being added
        activation = activations[i]
        current_padding = paddings[min(i, len(paddings)-1)] # Use last padding if list is short

        logging.debug(f"Adding Layer {i}: Type={layer_type}, Units/Filters={layer_units_or_filters}, Activation={activation}")

        # --- Input Layer Handling ---
        is_first_layer = (i == 0)
        layer_input_shape = input_shape if is_first_layer else None
        layer_input_dim = {'input_dim': input_dim_for_dense} if is_first_layer and input_dim_for_dense is not None and not input_shape else {}


        # --- Add Layers based on Type ---
        if layer_type == 'dense':
            if is_first_layer and layer_input_shape:
                 # Keras prefers input_shape for Dense if available, even if 1D
                 model.add(tf.keras.layers.Dense(layer_units_or_filters, activation=activation, input_shape=layer_input_shape, name=f"dense_{i}"))
            elif is_first_layer and layer_input_dim:
                 model.add(tf.keras.layers.Dense(layer_units_or_filters, activation=activation, **layer_input_dim, name=f"dense_{i}"))
            elif is_first_layer:
                 raise ValueError("Input shape/dimension must be specified for the first Dense layer (via --input-shape or inferred from --layers).")
            else:
                model.add(tf.keras.layers.Dense(layer_units_or_filters, activation=activation, name=f"dense_{i}"))

        elif layer_type in ['conv1d', 'conv2d']:
            if not is_first_layer and not model.layers: # Should not happen if logic is correct
                 raise ValueError(f"Cannot add {layer_type} layer: No preceding layer or input shape defined.")
            if is_first_layer and not layer_input_shape:
                raise ValueError(f"Input shape must be specified via --input-shape for the first {layer_type} layer.")

            kernel_size = kernel_sizes[min(kernel_idx, len(kernel_sizes)-1)] if kernel_sizes else 3 # Default kernel size
            stride_val = strides[min(stride_idx, len(strides)-1)] if strides else 1 # Default stride

            conv_params = {
                "filters": layer_units_or_filters,
                "kernel_size": kernel_size,
                "activation": activation,
                "strides": stride_val,
                "padding": current_padding,
                "name": f"{layer_type}_{i}"
            }
            if is_first_layer:
                conv_params["input_shape"] = layer_input_shape

            if layer_type == 'conv1d':
                model.add(tf.keras.layers.Conv1D(**conv_params))
            else: # conv2d
                model.add(tf.keras.layers.Conv2D(**conv_params))

            kernel_idx += 1
            stride_idx += 1 # Assuming strides apply to conv layers too

        elif layer_type in ['maxpool1d', 'maxpool2d', 'averagepool1d', 'averagepool2d']:
             if not model.layers:
                 raise ValueError(f"Cannot add {layer_type} layer: No preceding layer found.")

             pool_size = pool_sizes[min(pool_idx, len(pool_sizes)-1)] if pool_sizes else 2 # Default pool size
             stride_val = strides[min(stride_idx, len(strides)-1)] if strides else pool_size # Default stride matches pool size
             pool_params = {
                 "pool_size": pool_size,
                 "strides": stride_val,
                 "padding": current_padding,
                 "name": f"{layer_type}_{i}"
             }

             if 'maxpool1d' in layer_type: model.add(tf.keras.layers.MaxPooling1D(**pool_params))
             elif 'maxpool2d' in layer_type: model.add(tf.keras.layers.MaxPooling2D(**pool_params))
             elif 'averagepool1d' in layer_type: model.add(tf.keras.layers.AveragePooling1D(**pool_params))
             elif 'averagepool2d' in layer_type: model.add(tf.keras.layers.AveragePooling2D(**pool_params))

             pool_idx += 1
             stride_idx += 1

        elif layer_type == 'flatten':
            if not model.layers:
                 raise ValueError("Cannot add Flatten layer: No preceding layer found.")
            model.add(tf.keras.layers.Flatten(name=f"flatten_{i}"))

        elif layer_type == 'dropout':
             # Dropout rate could be specified in layers_str e.g., 'dropout:0.5'
             try:
                 rate = float(layer_configs[i+1]) # Use the value from layers_str for rate
             except ValueError:
                 rate = 0.5 # Default dropout rate
                 logging.warning(f"Could not parse dropout rate from '{layer_configs[i+1]}', using default {rate}")
             model.add(tf.keras.layers.Dropout(rate, name=f"dropout_{i}"))

        # Add more layer types here (e.g., LSTM, GRU, BatchNormalization) as needed

        else:
            raise ValueError(f"Unsupported layer type: '{layer_type}'")

    # --- Compile and Save ---
    logging.info("Compiling model...")
    try:
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy']) # Add default accuracy metric
    except Exception as e:
        logging.error(f"Error during model compilation: {e}")
        logging.error("Please ensure optimizer and loss function names are valid Keras identifiers.")
        raise

    logging.info("Model Summary:")
    model.summary(print_fn=logging.info) # Print summary to log

    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save using the recommended .keras format
        if not output_path.endswith(".keras"):
             output_path += ".keras"
             logging.warning(f"Appending '.keras' extension. Saving model to: {output_path}")
        model.save(output_path)
        logging.info(f"Model successfully created and saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving model to {output_path}: {e}")
        raise

# Example usage:
# create_network(layers_str="784,128,10", layer_types_str=None, activations_str="relu,softmax", ...) # Dense MNIST
# create_network(layers_str="None,32,64,10", layer_types_str="conv2d,conv2d,flatten,dense", activations_str="relu,relu,relu,softmax", input_shape_str="28,28,1", ...) # CNN MNIST
