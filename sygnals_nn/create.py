import tensorflow as tf

def create_network(layers, activation, loss, optimizer, output):
    """
    Create a neural network with the specified parameters and save the architecture.
    """
    # Parse layers and activation functions
    layers = [int(x) for x in layers.split(",")]
    activations = activation.split(",")

    if len(activations) == 1:
        activations = activations * (len(layers) - 1)  # Use the same activation for all layers

    if len(activations) != len(layers) - 1:
        raise ValueError("Number of activation functions must match the number of layers - 1.")

    # Build the model
    model = tf.keras.Sequential()
    for i in range(len(layers) - 1):
        if i == 0:  # First layer needs input_shape
            model.add(tf.keras.layers.Dense(layers[i + 1], activation=activations[i], input_shape=(layers[i],)))
        else:
            model.add(tf.keras.layers.Dense(layers[i + 1], activation=activations[i]))

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss)

    # Save the model to the specified output file
    model.save(output.replace(".h5", ".keras"))  # Save the entire model (architecture + weights + optimizer state)
    print(f"Model saved to {output}")
