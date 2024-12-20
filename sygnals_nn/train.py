import tensorflow as tf
from sygnals_nn.utils import load_data

def train_model(model_file, data_file, epochs, batch_size, learning_rate, inputs, outputs):
    """
    Train a neural network using the specified dataset.
    """
    # Step 1: Load the model from file
    model = tf.keras.models.load_model(model_file)  # Load the full model (architecture + weights + optimizer)

    model.summary()

    # Step 2: Load the training data
    X_train, Y_train = load_data(data_file, inputs, outputs)

    # Step 3: Recompile the model with the specified learning rate (if needed)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",  # Loss function for binary classification
        metrics=["accuracy"]
    )

    # Step 4: Train the model
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

    # Save the trained model back to the file
    model.save(model_file.replace(".h5", ".keras"))
    print(f"Trained model saved to {model_file.replace('.h5', '.keras')}")
