import pytest
import os
import pandas as pd
import json
from sygnals_nn.export import export_results
import numpy as np # Import numpy for comparison

# Fixture to create a dummy predictions file
@pytest.fixture
def predictions_file(tmp_path):
    """Creates a dummy CSV predictions file."""
    pred_path = tmp_path / "predictions_input.csv"
    # Simulate multi-column output (e.g., probabilities for multi-class)
    pd.DataFrame([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.6, 0.3, 0.1]
    ]).to_csv(pred_path, index=False, header=False)
    return pred_path

# Fixture for output file path
@pytest.fixture
def output_file(tmp_path):
    """Provides a path for the exported file."""
    return tmp_path / "exported_data" # Extension added by function

# --- Test Different Formats ---
def test_export_results_csv_to_csv(predictions_file, output_file):
    """Test exporting predictions from CSV to CSV format."""
    output_csv = str(output_file) + ".csv"
    export_results(str(predictions_file), "csv", output_csv)
    assert os.path.exists(output_csv)

    # Read back and check content/shape
    df_in = pd.read_csv(predictions_file, header=None)
    df_out = pd.read_csv(output_csv, header=None)
    pd.testing.assert_frame_equal(df_in, df_out)

def test_export_results_csv_to_json(predictions_file, output_file):
    """Test exporting predictions from CSV to JSON format."""
    output_json = str(output_file) + ".json"
    export_results(str(predictions_file), "json", output_json)
    assert os.path.exists(output_json)

    # Read back JSON and check content
    with open(output_json, 'r') as f:
        json_data = json.load(f)

    df_in = pd.read_csv(predictions_file, header=None)
    # Convert input df to list of lists for comparison
    expected_data = df_in.values.tolist()

    assert isinstance(json_data, list)
    assert len(json_data) == len(expected_data)
    # Check the values row by row, allowing for float precision
    for i in range(len(expected_data)):
         # Compare elements within each sublist, allowing for tolerance
         np.testing.assert_allclose(json_data[i], expected_data[i], rtol=1e-6)


def test_export_results_csv_to_raw(predictions_file, output_file):
    """Test exporting predictions from CSV to raw text format (plain CSV)."""
    output_raw = str(output_file) + ".txt" # Keep .txt extension for clarity if desired
    export_results(str(predictions_file), "raw", output_raw)
    assert os.path.exists(output_raw)

    # Read back the 'raw' output file (which is now CSV) and compare
    df_in = pd.read_csv(predictions_file, header=None)
    df_out = pd.read_csv(output_raw, header=None) # Read the output as CSV

    # Compare DataFrames directly
    pd.testing.assert_frame_equal(df_in, df_out)


# --- Test Error Handling ---
def test_export_results_unsupported_format(predictions_file, output_file):
    """Test error when an unsupported format is requested."""
    with pytest.raises(ValueError, match="Unsupported format"):
        export_results(str(predictions_file), "xml", str(output_file))

def test_export_results_input_not_found(output_file):
     """Test error when the input predictions file doesn't exist."""
     # This might raise FileNotFoundError in pd.read_csv inside the function
     with pytest.raises(FileNotFoundError):
         export_results("non_existent_preds.csv", "csv", str(output_file))
