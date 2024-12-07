import click
import os

from sygnals_nn.create import create_network
from sygnals_nn.train import train_model
from sygnals_nn.run import run_inference
from sygnals_nn.export import export_results

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

@click.group()
def cli():
    """A CLI tool for creating, training, and running neural networks."""
    pass

@cli.command()
@click.option('--layers', type=str, required=True, help="Comma-separated list of neurons per layer (e.g., '16,32,1').")
@click.option('--activation', type=str, default="relu", help="Comma-separated activation functions (e.g., 'relu,sigmoid').")
@click.option('--loss', type=str, default="binary_crossentropy", help="Loss function to use.")
@click.option('--optimizer', type=str, default="adam", help="Optimizer to use.")
@click.option('--output', type=str, required=True, help="Output file to save the model architecture.")
def create(layers, activation, loss, optimizer, output):
    """Create a neural network."""
    create_network(layers, activation, loss, optimizer, output)

@cli.command()
@click.option('--model', type=str, required=True, help="Path to the model file.")
@click.option('--data', type=str, required=True, help="Training data file (CSV, JSON, or raw text).")
@click.option('--epochs', type=int, default=100, help="Number of epochs.")
@click.option('--batch-size', type=int, default=32, help="Batch size.")
@click.option('--learning-rate', type=float, default=0.01, help="Learning rate.")
def train(model, data, epochs, batch_size, learning_rate):
    """Train a neural network."""
    print(f"DEBUG: model = {model}")
    train_model(model, data, epochs, batch_size, learning_rate)

@cli.command()
@click.option('--model', type=str, required=True, help="Path to the model file.")
@click.option('--input', type=str, required=True, help="Input data file (CSV, JSON, or raw text).")
@click.option('--output', type=str, default=None, help="Output file to save predictions.")
def run(model, input, output):
    """Run inference using a trained neural network."""
    run_inference(model, input, output)

@cli.command()
@click.option('--predictions', type=str, required=True, help="File containing predictions.")
@click.option('--format', type=str, default="csv", help="Output format (csv, json, or raw).")
@click.option('--output', type=str, required=True, help="File to save the exported results.")
def export(predictions, format, output):
    """Export predictions to a file."""
    export_results(predictions, format, output)

if __name__ == "__main__":
    cli()
