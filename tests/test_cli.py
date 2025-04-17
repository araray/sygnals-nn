import os
import pytest
from click.testing import CliRunner
from sygnals_nn.cli import cli
import pandas as pd
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Fixture for the Click CLI runner
@pytest.fixture
def runner():
    """Fixture to provide a Click test runner."""
    return CliRunner()

# Fixture to create dummy data files
@pytest.fixture
def create_files(tmp_path):
    """Fixture to create temporary data and model files."""
    # --- CSV Data ---
    csv_train_path = tmp_path / "train.csv"
    pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0],
        'feature2': [0.5, 0.6, 0.7, 0.8],
        'text': ['good one', 'bad one', 'good again', 'terrible one'],
        'label': [1, 0, 1, 0]
    }).to_csv(csv_train_path, index=False)

    csv_infer_path = tmp_path / "infer.csv"
    pd.DataFrame({
        'feature1': [5.0, 6.0],
        'feature2': [0.9, 1.0],
        'text': ['another good', 'another bad']
        # No label column for inference
    }).to_csv(csv_infer_path, index=False)

    # --- JSON Data ---
    json_train_path = tmp_path / "train.json"
    json_data = [
        {'id': 1, 'data_features': [1.1, 0.6], 'text_content': 'nice json', 'category': 'A'},
        {'id': 2, 'data_features': [2.1, 0.5], 'text_content': 'bad json', 'category': 'B'},
        {'id': 3, 'data_features': [3.1, 0.7], 'text_content': 'good json', 'category': 'A'},
    ]
    with open(json_train_path, 'w') as f:
        json.dump(json_data, f)

    json_infer_path = tmp_path / "infer.json"
    json_infer_data = [
         {'id': 4, 'data_features': [4.1, 0.9], 'text_content': 'test json'},
         {'id': 5, 'data_features': [5.1, 0.8], 'text_content': 'infer json'},
    ]
    with open(json_infer_path, 'w') as f:
        json.dump(json_infer_data, f)


    # --- Dummy Model (created later in tests) ---
    keras_model_path = tmp_path / "model.keras"
    onnx_model_path = tmp_path / "model.onnx"

    # --- Dummy Preprocessor (created later in tests) ---
    preprocessor_path = tmp_path / "preprocessor.joblib"

    return {
        "csv_train": csv_train_path,
        "csv_infer": csv_infer_path,
        "json_train": json_train_path,
        "json_infer": json_infer_path,
        "keras_model": keras_model_path,
        "onnx_model": onnx_model_path,
        "preprocessor": preprocessor_path,
        "processed_data": tmp_path / "processed.csv",
        "predictions": tmp_path / "predictions.csv"
    }

# --- Test Basic CLI ---
def test_cli_help(runner):
    """Test that the main CLI command `sygnals-nn --help` prints usage information."""
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert "Usage: sygnals-nn [OPTIONS] COMMAND [ARGS]..." in result.output
    assert "Commands:" in result.output
    assert "create" in result.output
    assert "train" in result.output
    assert "run" in result.output
    assert "preprocess" in result.output # Check new command
    assert "convert" in result.output    # Check new command
    assert "export" in result.output

# --- Test Create Command ---
def test_cli_create_dense_success(runner, create_files):
    """Test successful creation of a simple Dense model."""
    model_file = create_files["keras_model"]
    result = runner.invoke(cli, [
        'create',
        '--layers', '2,8,1', # Input dim 2, 8 hidden, 1 output
        '--activation', 'relu,sigmoid',
        '--loss', 'binary_crossentropy',
        '--optimizer', 'adam',
        '--output', str(model_file)
    ])
    print(f"Create Dense Output:\n{result.output}")
    assert result.exit_code == 0
    assert "Model successfully created and saved" in result.output
    assert model_file.exists()
    # Add check for model loading if needed (more involved)

def test_cli_create_cnn_success(runner, create_files):
    """Test successful creation of a simple Conv1D model."""
    model_file = create_files["keras_model"]
    result = runner.invoke(cli, [
        'create',
        '--layers', 'None,32,64,1', # Filters, output units
        '--layer-types', 'conv1d,flatten,dense',
        '--activation', 'relu,relu,sigmoid',
        '--input-shape', '10,1', # Example: 10 timesteps, 1 feature
        '--kernel-sizes', '3',
        '--loss', 'binary_crossentropy',
        '--optimizer', 'adam',
        '--output', str(model_file)
    ])
    print(f"Create CNN Output:\n{result.output}")
    assert result.exit_code == 0
    assert "Model successfully created and saved" in result.output
    assert model_file.exists()

def test_cli_create_missing_args(runner):
    """Test create command fails with missing required arguments."""
    result = runner.invoke(cli, ['create', '--layers', '2,1'])
    assert result.exit_code != 0
    assert "Error: Missing option '--output'" in result.output

# --- Test Preprocess Command ---
def test_cli_preprocess_tfidf_success(runner, create_files):
    """Test successful TF-IDF preprocessing."""
    result = runner.invoke(cli, [
        'preprocess',
        '--input-data', str(create_files["csv_train"]),
        '--output-data', str(create_files["processed_data"]),
        '--output-preprocessor', str(create_files["preprocessor"]),
        '--method', 'tfidf',
        '--text-col', 'text' # Use column name
    ])
    print(f"Preprocess TFIDF Output:\n{result.output}")
    assert result.exit_code == 0
    assert "Applying TF-IDF" in result.output
    assert "Processed data saved successfully" in result.output
    assert "Preprocessor object saved successfully" in result.output
    assert create_files["processed_data"].exists()
    assert create_files["preprocessor"].exists()

def test_cli_preprocess_scale_success(runner, create_files):
    """Test successful scaling preprocessing."""
    result = runner.invoke(cli, [
        'preprocess',
        '--input-data', str(create_files["csv_train"]),
        '--output-data', str(create_files["processed_data"]),
        '--output-preprocessor', str(create_files["preprocessor"]),
        '--method', 'scale',
        '--feature-cols', '0,feature2' # Use index and name
    ])
    print(f"Preprocess Scale Output:\n{result.output}")
    assert result.exit_code == 0
    assert "Applying StandardScaler" in result.output
    assert create_files["processed_data"].exists()
    assert create_files["preprocessor"].exists()

def test_cli_preprocess_missing_args(runner, create_files):
    """Test preprocess command fails with missing required arguments."""
    result = runner.invoke(cli, ['preprocess', '--input-data', str(create_files["csv_train"])])
    assert result.exit_code != 0
    assert "Error: Missing option" in result.output # General check

# --- Test Train Command ---
def test_cli_train_csv_success(runner, create_files):
    """Test successful training with CSV data."""
    # First, create a model
    model_file = create_files["keras_model"]
    runner.invoke(cli, ['create', '--layers', '2,4,1', '--activation', 'relu,sigmoid', '--output', str(model_file)])
    assert model_file.exists()

    result = runner.invoke(cli, [
        'train',
        '--model', str(model_file),
        '--data', str(create_files["csv_train"]),
        '--input-cols', 'feature1,feature2', # Use names
        '--label-cols', 'label',
        '--epochs', '1',
        '--batch-size', '2'
    ])
    print(f"Train CSV Output:\n{result.output}")
    assert result.exit_code == 0
    assert "Starting training process" in result.output
    assert "Epoch 1/1" in result.output
    assert "Trained Keras model saved successfully" in result.output

def test_cli_train_json_success(runner, create_files):
    """Test successful training with JSON data."""
    # Create a model compatible with json data features (2 inputs)
    model_file = create_files["keras_model"]
    runner.invoke(cli, ['create', '--layers', '2,4,1', '--activation', 'relu,sigmoid', '--output', str(model_file)])
    assert model_file.exists()

    # Need label encoding first for JSON category 'A'/'B'
    runner.invoke(cli, [
        'preprocess', '--input-data', str(create_files["json_train"]),
        '--output-data', str(create_files["processed_data"]), # Output encoded labels
        '--output-preprocessor', str(create_files["preprocessor"]), # Save encoder
        '--method', 'label_encode', '--label-col', 'category', '--json-label-key', 'category'
    ])
    assert create_files["processed_data"].exists() # Contains encoded labels

    # Manually create combined training data (features from json + encoded labels)
    # In a real scenario, preprocess script might handle this better
    with open(create_files["json_train"]) as f:
        json_feat_data = json.load(f)
    features = [item['data_features'] for item in json_feat_data]
    labels_df = pd.read_csv(create_files["processed_data"]) # Read encoded labels
    combined_df = pd.DataFrame(features, columns=['f1', 'f2'])
    combined_df['label'] = labels_df['category'] # Add encoded labels
    combined_train_path = create_files["keras_model"].parent / "combined_train.csv"
    combined_df.to_csv(combined_train_path, index=False)


    result = runner.invoke(cli, [
        'train',
        '--model', str(model_file),
        '--data', str(combined_train_path), # Use combined data
        '--input-cols', 'f1,f2',
        '--label-cols', 'label',
        '--epochs', '1'
        # No need for json keys here as we load combined CSV
    ])
    print(f"Train JSON (Combined) Output:\n{result.output}")
    assert result.exit_code == 0
    assert "Starting training process" in result.output
    assert "Trained Keras model saved successfully" in result.output


def test_cli_train_with_onnx_export(runner, create_files):
    """Test training with direct ONNX export."""
    model_file = create_files["keras_model"]
    onnx_file = create_files["onnx_model"]
    runner.invoke(cli, ['create', '--layers', '2,4,1', '--activation', 'relu,sigmoid', '--output', str(model_file)])

    result = runner.invoke(cli, [
        'train',
        '--model', str(model_file),
        '--data', str(create_files["csv_train"]),
        '--input-cols', '0,1', # Use indices
        '--label-cols', '3',
        '--epochs', '1',
        '--export-onnx', str(onnx_file) # Add export flag
    ])
    print(f"Train with ONNX Export Output:\n{result.output}")
    assert result.exit_code == 0
    assert "Trained Keras model saved successfully" in result.output
    assert "Exporting trained model to ONNX format" in result.output
    assert "Model successfully exported to ONNX" in result.output
    assert onnx_file.exists()


# --- Test Convert Command ---
def test_cli_convert_success(runner, create_files):
    """Test successful Keras to ONNX conversion."""
    model_file = create_files["keras_model"]
    onnx_file = create_files["onnx_model"]
    runner.invoke(cli, ['create', '--layers', '2,4,1', '--activation', 'relu,sigmoid', '--output', str(model_file)])

    result = runner.invoke(cli, [
        'convert',
        '--keras-model', str(model_file),
        '--output-onnx', str(onnx_file)
    ])
    print(f"Convert Output:\n{result.output}")
    assert result.exit_code == 0
    assert "Starting ONNX conversion" in result.output
    assert "Model successfully converted and saved to ONNX" in result.output
    assert onnx_file.exists()

# --- Test Run Command ---
def test_cli_run_keras_success(runner, create_files):
    """Test successful inference with a Keras model."""
    model_file = create_files["keras_model"]
    runner.invoke(cli, ['create', '--layers', '2,4,1', '--activation', 'relu,sigmoid', '--output', str(model_file)])
    # Dummy training needed to ensure model is built and saved correctly after compilation
    runner.invoke(cli, ['train', '--model', str(model_file), '--data', str(create_files["csv_train"]), '--input-cols', '0,1', '--label-cols', '3', '--epochs', '1'])


    result = runner.invoke(cli, [
        'run',
        '--model', str(model_file),
        '--input-data', str(create_files["csv_infer"]),
        '--input-cols', 'feature1,feature2',
        '--output', str(create_files["predictions"])
    ])
    print(f"Run Keras Output:\n{result.output}")
    assert result.exit_code == 0
    assert "Running inference with model" in result.output
    assert "Performing Keras inference" in result.output
    assert "Predictions successfully saved" in result.output
    assert create_files["predictions"].exists()
    # Check number of predictions
    preds = pd.read_csv(create_files["predictions"], header=None)
    infer_data = pd.read_csv(create_files["csv_infer"])
    assert len(preds) == len(infer_data)


def test_cli_run_onnx_success(runner, create_files):
    """Test successful inference with an ONNX model."""
    # Create Keras model
    model_file = create_files["keras_model"]
    runner.invoke(cli, ['create', '--layers', '2,4,1', '--activation', 'relu,sigmoid', '--output', str(model_file)])
    # Dummy training
    runner.invoke(cli, ['train', '--model', str(model_file), '--data', str(create_files["csv_train"]), '--input-cols', '0,1', '--label-cols', '3', '--epochs', '1'])
    # Convert to ONNX
    onnx_file = create_files["onnx_model"]
    runner.invoke(cli, ['convert', '--keras-model', str(model_file), '--output-onnx', str(onnx_file)])
    assert onnx_file.exists()

    result = runner.invoke(cli, [
        'run',
        '--model', str(onnx_file), # Use ONNX model
        '--input-data', str(create_files["csv_infer"]),
        '--input-cols', '0,1', # Indices
        '--output', str(create_files["predictions"])
    ])
    print(f"Run ONNX Output:\n{result.output}")
    # Check if ONNX runtime is available, skip if not
    try:
        import onnxruntime
        assert result.exit_code == 0
        assert "Running inference with model" in result.output
        assert "Performing ONNX inference" in result.output
        assert "Predictions successfully saved" in result.output
        assert create_files["predictions"].exists()
        preds = pd.read_csv(create_files["predictions"], header=None)
        infer_data = pd.read_csv(create_files["csv_infer"])
        assert len(preds) == len(infer_data)
    except ImportError:
        pytest.skip("onnxruntime not installed, skipping ONNX run test")
    except Exception as e:
         # Catch other potential ONNX runtime errors during test
         pytest.fail(f"ONNX run failed unexpectedly: {e}\nOutput:\n{result.output}")


def test_cli_run_with_preprocessor_success(runner, create_files):
    """Test successful inference using a saved preprocessor."""
     # 1. Preprocess text data to create vectorizer
    vectorizer_path = create_files["preprocessor"]
    runner.invoke(cli, [
        'preprocess', '--input-data', str(create_files["csv_train"]),
        '--output-data', str(create_files["processed_data"]), # Dummy output data
        '--output-preprocessor', str(vectorizer_path),
        '--method', 'tfidf', '--text-col', 'text'
    ])
    assert vectorizer_path.exists()

    # 2. Create a model compatible with TF-IDF output dimension
    #    Need to know the dimension first - load the vectorizer
    vectorizer = joblib.load(vectorizer_path)
    n_features = len(vectorizer.vocabulary_)
    model_file = create_files["keras_model"]
    runner.invoke(cli, ['create', '--layers', f'{n_features},16,1', '--activation', 'relu,sigmoid', '--output', str(model_file)])
    # Dummy training
    # We would need to train on the *processed* data corresponding to csv_train
    # For simplicity of CLI test, we skip actual training here, just ensure run uses preprocessor

    # 3. Run inference, providing the *original* text column and the preprocessor
    result = runner.invoke(cli, [
        'run',
        '--model', str(model_file),
        '--input-data', str(create_files["csv_infer"]), # Raw inference data
        '--input-cols', 'text', # Specify the *original* text column
        '--preprocessor-path', str(vectorizer_path), # Provide the preprocessor
        '--output', str(create_files["predictions"])
    ])
    print(f"Run with Preprocessor Output:\n{result.output}")
    assert result.exit_code == 0
    assert "Using preprocessor" in result.output
    assert "Applying preprocessor to column: text" in result.output # Check log message from utils
    assert "Predictions successfully saved" in result.output
    assert create_files["predictions"].exists()


# --- Test Export Command ---
def test_cli_export_success(runner, create_files):
    """Test successful export of predictions."""
    # Create dummy predictions file
    preds_path = create_files["predictions"]
    pd.DataFrame({'pred': [0.1, 0.9, 0.5]}).to_csv(preds_path, index=False, header=False)

    export_file = create_files["keras_model"].parent / "exported_preds.json"

    result = runner.invoke(cli, [
        'export',
        '--predictions', str(preds_path),
        '--format', 'json',
        '--output', str(export_file)
    ])
    print(f"Export Output:\n{result.output}")
    assert result.exit_code == 0
    assert "Predictions exported to" in result.output
    assert export_file.exists()
    # Add check for content if needed
