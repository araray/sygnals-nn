import tensorflow as tf
from sygnals_nn.utils import load_data

def run_inference(model_file, input_file, output_file):
    """
    Run inference using a trained neural network.
    """
    # Load the trained model
    model = tf.keras.models.load_model(model_file)

    # Load the input data
    X_input, _ = load_data(input_file)

    # Perform inference
    predictions = model.predict(X_input)

    # Save predictions
    if output_file:
        import pandas as pd
        pd.DataFrame(predictions).to_csv(output_file, index=False, header=False)
        print(f"Predictions saved to {output_file}")
    else:
        print(predictions)
