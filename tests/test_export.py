import pytest
import os
import pandas as pd
import json
from sygnals_nn.export import export_results

# Fixture to create a dummy predictions file
@pytest.fixture
def predictions_file(tmp_path):
    """Creates a dummy CSV predictions file."""
    pred_path = tmp_path / "predictions_input.csv"
    # Simulate multi-column output (e.g., probabilities for multi-class)
    pd.DataFrame({
        'p0': [0.8, 0.1, 0.6],
        'p1': [0.1, 0.7, 0.3],
        'p2': [0.1, 0.2, 0.1]
    }).to_csv(pred_path, index=False, header=False)
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
    # JSON export might store as dicts {col_idx: value} or just lists
    # tf2onnx default is 'records' which is list of dicts if columns exist,
    # but here we read with header=None, so it becomes list of lists.
    # Let's check the values row by row
    for i in range(len(expected_data)):
         # Handle potential type differences (e.g., int vs float) if necessary
         assert json_data[i] == expected_data[i]


def test_export_results_csv_to_raw(predictions_file, output_file):
    """Test exporting predictions from CSV to raw text format."""
    output_raw = str(output_file) + ".txt" # Assume raw saves as .txt
    export_results(str(predictions_file), "raw", output_raw)
    assert os.path.exists(output_raw)

    # Read back raw text and check content
    with open(output_raw, 'r') as f:
        lines = f.readlines()

    df_in = pd.read_csv(predictions_file, header=None)
    assert len(lines) == df_in.shape[0] # Check number of lines
    # Check first line content (adjust based on pandas to_string format)
    expected_first_line = ",".join(map(str, df_in.iloc[0].values)) + "\n" # Simple expectation
    # Pandas to_string might have different spacing, more robust check needed if format is strict
    assert lines[0].strip().replace(" ", "") == expected_first_line.strip().replace(" ", "")


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
