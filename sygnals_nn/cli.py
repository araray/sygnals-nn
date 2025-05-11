import click
import os

# Import functions from other modules
from sygnals_nn.create import create_network
from sygnals_nn.train import train_model
from sygnals_nn.run import run_inference
from sygnals_nn.export import export_results
# Import new modules/functions
from sygnals_nn.convert import convert_to_onnx
from sygnals_nn.preprocess import preprocess_data

# Suppress TensorFlow informational messages (optional)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- Main CLI Group ---
@click.group()
def cli():
    """
    Sygnals-NN: A CLI tool for creating, training, running, and managing
    neural networks, with support for various data formats, preprocessing,
    and probabilistic modeling.
    """
    pass

# --- Create Command ---
@cli.command()
@click.option('--layers', type=str, required=True,
              help="Comma-separated list of neurons/filters per layer (e.g., 'input_dim,64,32,output_dim' or 'input_shape,conv1d:32,flatten,dense:10'). For probabilistic regression, the output_dim is the number of target dimensions (e.g., 1 for univariate regression).")
@click.option('--layer-types', type=str, default=None,
              help="Comma-separated list of layer types corresponding to --layers (e.g., 'dense,dense,dense' or 'conv1d,maxpool1d,flatten,dense'). Required if using non-dense layers.")
@click.option('--activation', type=str, default="relu",
              help="Comma-separated activation functions for layers (e.g., 'relu,relu,sigmoid'). Applied to Dense/Conv layers. If single, applied to all. For probabilistic Gaussian output, the final activation applies to the mean; log-variance is linear.")
@click.option('--loss', type=str, default="binary_crossentropy", # This default will be overridden by train for probabilistic models
              help="Loss function for compiling the model (e.g., 'categorical_crossentropy', 'mse'). For probabilistic models, specific NLL losses are typically used during training (e.g., 'gaussian_nll').")
@click.option('--optimizer', type=str, default="adam",
              help="Optimizer for compiling the model (e.g., 'sgd', 'rmsprop').")
@click.option('--output', type=str, required=True,
              help="Output file path to save the created Keras model (.keras format).")
# New options for Probabilistic Models
@click.option('--model-type', type=click.Choice(['deterministic', 'probabilistic_regression', 'probabilistic_classification'], case_sensitive=False),
              default='deterministic', show_default=True,
              help="Type of model to create. 'probabilistic_regression' enables models that output distribution parameters.")
@click.option('--output-distribution', type=click.Choice(['gaussian', 'laplace', 'categorical'], case_sensitive=False),
              default=None, # Default depends on model_type; e.g., 'gaussian' for prob_regression
              help="Specifies the output probability distribution for probabilistic models (e.g., 'gaussian' for probabilistic_regression).")
# CNN specific options
@click.option('--kernel-sizes', type=str, default=None, help="Comma-separated kernel sizes for Conv layers (e.g., '3,3').")
@click.option('--pool-sizes', type=str, default=None, help="Comma-separated pool sizes for MaxPooling layers (e.g., '2,2').")
@click.option('--strides', type=str, default=None, help="Comma-separated strides for Conv/Pooling layers.")
@click.option('--padding', type=str, default='valid', help="Padding type for Conv/Pooling layers ('valid' or 'same'). Can be comma-separated.")
@click.option('--input-shape', type=str, default=None, help="Explicit input shape for the first layer, required for CNNs (e.g., 'timesteps,features' for Conv1D or 'height,width,channels' for Conv2D). Comma-separated.")
def create(layers, layer_types, activation, loss, optimizer, output,
           model_type, output_distribution, # New params
           kernel_sizes, pool_sizes, strides, padding, input_shape):
    """
    Create and save a neural network architecture (Keras model).
    Supports Dense layers by default, and other types like Conv1D/Conv2D,
    MaxPooling, Flatten via --layer-types and related options.
    Can also create models for probabilistic regression.
    """
    # Basic validation for probabilistic model options
    if model_type == 'probabilistic_regression' and not output_distribution:
        output_distribution = 'gaussian' # Default for probabilistic regression
        click.echo(f"Info: --output-distribution not specified for probabilistic_regression, defaulting to '{output_distribution}'.")
    elif model_type == 'deterministic' and output_distribution:
        click.echo("Warning: --output-distribution is specified but --model-type is 'deterministic'. Distribution will be ignored for model creation.")
    elif model_type == 'probabilistic_classification':
        # For now, probabilistic_classification might imply standard softmax output
        # but with different handling in train/run (e.g. MC Dropout or specific loss)
        if not output_distribution:
            output_distribution = 'categorical' # Default for probabilistic classification
            click.echo(f"Info: --output-distribution not specified for probabilistic_classification, defaulting to '{output_distribution}'.")


    create_network(
        layers_str=layers,
        layer_types_str=layer_types,
        activations_str=activation,
        loss=loss, # Loss specified here is for Keras model saving, actual training loss might differ
        optimizer=optimizer,
        output_path=output,
        model_type=model_type, # Pass new param
        output_distribution=output_distribution, # Pass new param
        kernel_sizes_str=kernel_sizes,
        pool_sizes_str=pool_sizes,
        strides_str=strides,
        padding_str=padding,
        input_shape_str=input_shape
    )

# --- Train Command ---
@cli.command()
@click.option('--model', type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to the Keras model file (.keras) to train.")
@click.option('--data', type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to the training data file (CSV or JSON).")
@click.option('--epochs', type=int, default=100, show_default=True,
              help="Number of training epochs.")
@click.option('--batch-size', type=int, default=32, show_default=True,
              help="Batch size for training.")
@click.option('--learning-rate', type=float, default=0.001, show_default=True,
              help="Learning rate for the optimizer.")
# Data loading options
@click.option('--input-cols', type=str, required=True,
              help="Comma-separated list of column indices or names for input features (e.g., '0,1,2,3' or 'feature1,feature2').")
@click.option('--label-cols', type=str, required=True,
              help="Comma-separated list of column indices or names for labels (e.g., '4' or 'target').")
@click.option('--json-input-key', type=str, default='features', show_default=True,
              help="Key in JSON objects containing the input features (if data is JSON). Assumes a list of objects.")
@click.option('--json-label-key', type=str, default='label', show_default=True,
              help="Key in JSON objects containing the label (if data is JSON).")
# ONNX export option
@click.option('--export-onnx', type=click.Path(dir_okay=False), default=None,
              help="Optional: Path to save the trained model in ONNX format after training.")
# Loss override for training - useful for probabilistic models where create might save with a placeholder
@click.option('--training-loss', type=str, default=None,
              help="Optional: Override the loss function specifically for this training run. E.g., 'gaussian_nll'. If not set, uses the loss the model was compiled with or infers for probabilistic models.")
def train(model, data, epochs, batch_size, learning_rate, input_cols, label_cols, json_input_key, json_label_key, export_onnx, training_loss):
    """
    Train a Keras model using the specified dataset.
    Loads data from CSV or JSON files based on column indices/names.
    Optionally exports the trained model to ONNX format.
    For probabilistic models, appropriate loss functions (e.g., NLL) will be used.
    """
    train_model(
        model_path=model,
        data_path=data,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        input_cols_str=input_cols,
        label_cols_str=label_cols,
        json_input_key=json_input_key,
        json_label_key=json_label_key,
        export_onnx_path=export_onnx,
        training_loss_override=training_loss # Pass the new option
    )

# --- Run Command ---
@cli.command()
@click.option('--model', type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to the trained Keras model file (.keras) or ONNX model file (.onnx).")
@click.option('--input-data', type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to the input data file for inference (CSV or JSON). Labels are ignored.")
@click.option('--output', type=click.Path(dir_okay=False), default=None,
              help="Optional: Output file path to save predictions (CSV format). If omitted, prints to console.")
# Data loading options
@click.option('--input-cols', type=str, required=True,
              help="Comma-separated list of column indices or names for input features (e.g., '0,1,2,3' or 'feature1,feature2').")
@click.option('--json-input-key', type=str, default='features', show_default=True,
              help="Key in JSON objects containing the input features (if data is JSON).")
# Preprocessing option
@click.option('--preprocessor-path', type=click.Path(exists=True, dir_okay=False), default=None,
              help="Optional: Path to a saved preprocessor object (e.g., TF-IDF vectorizer) to apply to input data before inference.")
# Probabilistic inference options
@click.option('--prediction-mode', type=click.Choice(['params', 'samples', 'mean_stddev'], case_sensitive=False),
              default='params', show_default=True, # Default might change based on model type in run.py
              help="For probabilistic models: 'params' outputs distribution parameters, 'samples' outputs multiple samples, 'mean_stddev' outputs mean and std deviation.")
@click.option('--mc-dropout', is_flag=True, default=False, show_default=True,
              help="Enable Monte Carlo Dropout for uncertainty estimation (if model has dropout layers).")
@click.option('--num-samples', type=int, default=30, show_default=True,
              help="Number of samples for MC Dropout or sampling from probabilistic model output.")
def run(model, input_data, output, input_cols, json_input_key, preprocessor_path,
        prediction_mode, mc_dropout, num_samples): # New params
    """
    Run inference using a trained Keras or ONNX model.
    Loads data, optionally applies preprocessing, makes predictions,
    and saves or prints the results. Supports probabilistic model outputs.
    """
    run_inference(
        model_path=model,
        input_data_path=input_data,
        output_path=output,
        input_cols_str=input_cols,
        json_input_key=json_input_key,
        preprocessor_path=preprocessor_path,
        prediction_mode=prediction_mode, # Pass new param
        mc_dropout=mc_dropout, # Pass new param
        num_samples=num_samples # Pass new param
    )

# --- Convert Command (New) ---
@cli.command()
@click.option('--keras-model', type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to the trained Keras model file (.keras) to convert.")
@click.option('--output-onnx', type=click.Path(dir_okay=False), required=True,
              help="Output file path to save the converted ONNX model (.onnx).")
@click.option('--input-signature', type=str, default=None,
              help="Optional: Define the input signature for ONNX conversion. Example: '[tf.TensorSpec(shape=(None, num_features), dtype=tf.float32)]'. Automatically inferred if possible.")
def convert(keras_model, output_onnx, input_signature):
    """
    Convert a trained Keras model to ONNX format.
    """
    convert_to_onnx(
        keras_model_path=keras_model,
        output_onnx_path=output_onnx,
        input_signature_str=input_signature
    )

# --- Preprocess Command (New) ---
@cli.command()
@click.option('--input-data', type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to the raw input data file (CSV or JSON).")
@click.option('--output-data', type=click.Path(dir_okay=False), required=True,
              help="Path to save the preprocessed numerical data (CSV format).")
@click.option('--output-preprocessor', type=click.Path(dir_okay=False), required=True,
              help="Path to save the fitted preprocessor object (e.g., vectorizer state).")
@click.option('--method', type=click.Choice(['tfidf', 'count', 'scale', 'label_encode'], case_sensitive=False), required=True,
              help="Preprocessing method to apply.")
# Data selection options
@click.option('--text-col', type=str, default=None,
              help="Column index or name containing text data (required for 'tfidf', 'count').")
@click.option('--label-col', type=str, default=None,
              help="Column index or name containing labels to encode (required for 'label_encode').")
@click.option('--feature-cols', type=str, default=None,
              help="Comma-separated column indices or names for numerical features to scale (required for 'scale').")
# JSON specific options
@click.option('--json-text-key', type=str, default='text', show_default=True,
              help="Key in JSON objects for text data (if input is JSON).")
@click.option('--json-label-key', type=str, default='label', show_default=True,
              help="Key in JSON objects for label data (if input is JSON).")
@click.option('--json-feature-key', type=str, default='features', show_default=True,
              help="Key in JSON objects for numerical feature data (if input is JSON).")
# Method specific options (add more as needed)
@click.option('--tfidf-max-features', type=int, default=None, help="Maximum number of features for TF-IDF.")
def preprocess(input_data, output_data, output_preprocessor, method, text_col, label_col, feature_cols, json_text_key, json_label_key, json_feature_key, tfidf_max_features):
    """
    Preprocess raw data (CSV or JSON) using specified methods.
    Applies transformations like TF-IDF, scaling, or label encoding,
    saving both the transformed data and the preprocessor state.
    """
    preprocess_data(
        input_path=input_data,
        output_data_path=output_data,
        output_preprocessor_path=output_preprocessor,
        method=method,
        text_col=text_col,
        label_col=label_col,
        feature_cols_str=feature_cols,
        json_text_key=json_text_key,
        json_label_key=json_label_key,
        json_feature_key=json_feature_key,
        tfidf_max_features=tfidf_max_features
        # Add other method-specific parameters here
    )


# --- Export Command (Kept for compatibility, might be less needed if run saves directly) ---
@cli.command()
@click.option('--predictions', type=click.Path(exists=True, dir_okay=False), required=True,
              help="File containing predictions (usually CSV from 'run' command).")
@click.option('--format', type=click.Choice(['csv', 'json', 'raw'], case_sensitive=False), default="csv", show_default=True,
              help="Output format for exporting predictions.")
@click.option('--output', type=click.Path(dir_okay=False), required=True,
              help="File path to save the exported results.")
def export(predictions, format, output):
    """
    Export predictions from a file to a specified format.
    (Primarily useful if 'run' command didn't save directly).
    """
    export_results(predictions, format, output)

# --- Entry Point ---
if __name__ == "__main__":
    cli()
