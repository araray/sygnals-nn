import numpy as np
import pandas as pd
import pytest
import tempfile
import os
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sygnals_nn.utils import load_data, _parse_col_indices_or_names # Import helper for direct testing
import logging # Import logging

# Ensure logs are captured during testing
logging.basicConfig(level=logging.INFO)


# --- Fixture for creating temporary files ---
@pytest.fixture
def temp_files(tmp_path):
    """Creates various temporary data files."""
    files = {}

    # CSV with header
    csv_header_path = tmp_path / "data_header.csv"
    pd.DataFrame({
        'feat1': [0.1, 0.3, 0.5],
        'feat2': [0.2, 0.4, 0.6],
        'text': ['hello world', 'test data', 'another row'],
        'label': [1, 0, 1]
    }).to_csv(csv_header_path, index=False)
    files["csv_header"] = csv_header_path

    # CSV without header
    csv_no_header_path = tmp_path / "data_no_header.csv"
    pd.DataFrame([
        [1.1, 1.2, 0],
        [1.3, 1.4, 1],
        [1.5, 1.6, 0]
    ]).to_csv(csv_no_header_path, index=False, header=False)
    files["csv_no_header"] = csv_no_header_path

    # CSV for inference (only features)
    csv_infer_path = tmp_path / "data_infer.csv"
    pd.DataFrame({
        'feat1': [0.7, 0.9],
        'feat2': [0.8, 1.0],
        'text': ['infer one', 'infer two']
    }).to_csv(csv_infer_path, index=False)
    files["csv_infer"] = csv_infer_path

    # JSON - List of objects
    json_list_path = tmp_path / "data_list.json"
    json_list_data = [
        {'f1': 2.1, 'f2': 2.2, 'lbl': 0, 'txt': 'json one'},
        {'f1': 2.3, 'f2': 2.4, 'lbl': 1, 'txt': 'json two'}
    ]
    with open(json_list_path, 'w') as f: json.dump(json_list_data, f)
    files["json_list"] = json_list_path

    # JSON - Dict of lists
    json_dict_path = tmp_path / "data_dict.json"
    json_dict_data = {
        'f1': [3.1, 3.3],
        'f2': [3.2, 3.4],
        'lbl': [1, 0],
        'txt': ['dict one', 'dict two']
    }
    with open(json_dict_path, 'w') as f: json.dump(json_dict_data, f)
    files["json_dict"] = json_dict_path

    # JSON - Inference
    json_infer_path = tmp_path / "data_infer.json"
    json_infer_data = [
        {'f1': 4.1, 'f2': 4.2, 'txt': 'infer three'},
        {'f1': 4.3, 'f2': 4.4, 'txt': 'infer four'}
    ]
    with open(json_infer_path, 'w') as f: json.dump(json_infer_data, f)
    files["json_infer"] = json_infer_path

    # CSV with non-numeric label
    csv_bad_label_path = tmp_path / "data_bad_label.csv"
    pd.DataFrame({
        'f1': [1,2], 'f2': [3,4], 'label': [0, 'bad'] # 'bad' is non-numeric
    }).to_csv(csv_bad_label_path, index=False)
    files["csv_bad_label"] = csv_bad_label_path

    # CSV with non-numeric feature
    csv_bad_feat_path = tmp_path / "data_bad_feat.csv"
    pd.DataFrame({
        'f1': [1,'x'], 'f2': [3,4], 'label': [0, 1] # 'x' is non-numeric
    }).to_csv(csv_bad_feat_path, index=False)
    files["csv_bad_feat"] = csv_bad_feat_path

    return files

# --- Test _parse_col_indices_or_names helper ---
def test_parse_cols_indices():
    df_cols = pd.Index(['A', 'B', 'C', 'D'])
    assert _parse_col_indices_or_names(df_cols, "0,2") == ['A', 'C']
    assert _parse_col_indices_or_names(df_cols, " 3 , 1 ") == ['D', 'B'] # Test spacing

def test_parse_cols_names():
    df_cols = pd.Index(['feat1', 'feat2', 'label', 'id'])
    assert _parse_col_indices_or_names(df_cols, "feat1,label") == ['feat1', 'label']
    assert _parse_col_indices_or_names(df_cols, " id , feat2") == ['id', 'feat2']

def test_parse_cols_mixed():
    df_cols = pd.Index(['A', 'B', 'C', 'D'])
    assert _parse_col_indices_or_names(df_cols, "0, C, 1") == ['A', 'C', 'B']

def test_parse_cols_invalid_index():
    df_cols = pd.Index(['A', 'B'])
    # This assertion should now pass with the updated _parse_col_indices_or_names logic
    with pytest.raises(ValueError, match="Column index 2 is out of bounds"):
        _parse_col_indices_or_names(df_cols, "0,2")

def test_parse_cols_invalid_name():
    df_cols = pd.Index(['A', 'B'])
    with pytest.raises(ValueError, match="Column name 'C' not found"):
        _parse_col_indices_or_names(df_cols, "A,C")

def test_parse_cols_empty():
     df_cols = pd.Index(['A', 'B'])
     assert _parse_col_indices_or_names(df_cols, "") == []
     assert _parse_col_indices_or_names(df_cols, None) == []


# --- Test load_data ---

# CSV Tests
def test_load_data_csv_header_names(temp_files):
    """Load CSV with header using column names."""
    X, Y = load_data(temp_files["csv_header"], input_cols_str='feat1,feat2', label_cols_str='label')
    assert X.shape == (3, 2)
    assert Y.shape == (3, 1)
    np.testing.assert_array_almost_equal(X, [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    np.testing.assert_array_almost_equal(Y.flatten(), [1, 0, 1])
    assert X.dtype == np.float32
    assert Y.dtype == np.float32

def test_load_data_csv_header_indices(temp_files):
    """Load CSV with header using column indices."""
    X, Y = load_data(temp_files["csv_header"], input_cols_str='0,1', label_cols_str='3')
    assert X.shape == (3, 2)
    assert Y.shape == (3, 1)
    np.testing.assert_array_almost_equal(X, [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    np.testing.assert_array_almost_equal(Y.flatten(), [1, 0, 1])

def test_load_data_csv_no_header_indices(temp_files):
    """Load CSV without header using column indices."""
    X, Y = load_data(temp_files["csv_no_header"], input_cols_str='0,1', label_cols_str='2')
    assert X.shape == (3, 2) # Check shape is correct now
    assert Y.shape == (3, 1)
    np.testing.assert_array_almost_equal(X, [[1.1, 1.2], [1.3, 1.4], [1.5, 1.6]])
    np.testing.assert_array_almost_equal(Y.flatten(), [0, 1, 0])

def test_load_data_csv_inference(temp_files):
    """Load CSV for inference (no labels)."""
    X, Y = load_data(temp_files["csv_infer"], input_cols_str='feat1,feat2', is_inference=True)
    assert X.shape == (2, 2)
    assert Y is None
    np.testing.assert_array_almost_equal(X, [[0.7, 0.8], [0.9, 1.0]])

# JSON Tests
def test_load_data_json_list_keys(temp_files):
    """Load JSON list of objects using keys."""
    X, Y = load_data(temp_files["json_list"], input_cols_str='f1,f2', label_cols_str='lbl')
    assert X.shape == (2, 2)
    assert Y.shape == (2, 1)
    np.testing.assert_array_almost_equal(X, [[2.1, 2.2], [2.3, 2.4]])
    np.testing.assert_array_almost_equal(Y.flatten(), [0, 1])

def test_load_data_json_dict_keys(temp_files):
    """Load JSON dict of lists using keys."""
    X, Y = load_data(temp_files["json_dict"], input_cols_str='f1,f2', label_cols_str='lbl')
    assert X.shape == (2, 2)
    assert Y.shape == (2, 1)
    np.testing.assert_array_almost_equal(X, [[3.1, 3.2], [3.3, 3.4]])
    np.testing.assert_array_almost_equal(Y.flatten(), [1, 0])

def test_load_data_json_inference(temp_files):
    """Load JSON for inference."""
    X, Y = load_data(temp_files["json_infer"], input_cols_str='f1,f2', is_inference=True)
    assert X.shape == (2, 2)
    assert Y is None
    np.testing.assert_array_almost_equal(X, [[4.1, 4.2], [4.3, 4.4]])

# Preprocessor Tests
def test_load_data_with_scaler(temp_files, tmp_path):
    """Load data and apply a StandardScaler."""
    # Create and save a scaler
    scaler_path = tmp_path / "scaler.joblib"
    scaler = StandardScaler()
    df_train = pd.read_csv(temp_files["csv_header"])
    scaler.fit(df_train[['feat1', 'feat2']])
    joblib.dump(scaler, scaler_path)

    # Load inference data using the scaler
    X, Y = load_data(
        temp_files["csv_infer"],
        input_cols_str='feat1,feat2', # Specify columns to scale
        is_inference=True,
        preprocessor=scaler # Pass the loaded object directly (or path if run.py loads it)
    )
    assert X.shape == (2, 2)
    assert Y is None
    # Check if data is scaled (mean approx 0, std approx 1 relative to training data)
    # Calculate expected scaled values based on training data fit
    expected_X = scaler.transform(pd.read_csv(temp_files["csv_infer"])[['feat1', 'feat2']])
    np.testing.assert_array_almost_equal(X, expected_X)

def test_load_data_with_tfidf(temp_files, tmp_path):
    """Load data and apply a TfidfVectorizer."""
    # Create and save vectorizer
    vectorizer_path = tmp_path / "tfidf.joblib"
    vectorizer = TfidfVectorizer()
    df_train = pd.read_csv(temp_files["csv_header"])
    vectorizer.fit(df_train['text'])
    joblib.dump(vectorizer, vectorizer_path)
    n_features = len(vectorizer.vocabulary_)

    # Load inference data using the vectorizer
    X, Y = load_data(
        temp_files["csv_infer"],
        input_cols_str=None, # Input cols not needed when text_col_for_preprocess is set
        is_inference=True,
        preprocessor=vectorizer, # Pass the loaded object
        text_col_for_preprocess='text' # Specify the text column
    )
    assert X.shape == (2, n_features) # Rows from infer, cols from vocab
    assert Y is None
    assert X.dtype == np.float32 # Should be converted to float

# Error Handling Tests
def test_load_data_file_not_found():
    """Test error when file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv", "0", "1")

def test_load_data_missing_input_cols():
    """Test error if input columns are required but not provided."""
    # Create dummy file
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w', delete=False) as f:
        f.write("a,b,c\n1,2,0")
        tmp_filename = f.name
    # Expect error because neither input_cols_str nor preprocessor+text_col specified
    with pytest.raises(ValueError, match="Input columns must be specified"):
        load_data(tmp_filename, input_cols_str=None, label_cols_str='c')
    os.remove(tmp_filename)


def test_load_data_missing_label_cols_for_train():
    """Test error if label columns are required but not provided during training."""
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w', delete=False) as f:
        f.write("a,b,c\n1,2,0")
        tmp_filename = f.name
    with pytest.raises(ValueError, match="Label columns must be specified"):
        load_data(tmp_filename, input_cols_str='a,b', label_cols_str=None, is_inference=False)
    os.remove(tmp_filename)

def test_load_data_bad_label_type(temp_files):
    """Test error when label column contains non-numeric data."""
    # FIX: Update regex to match the error raised by astype(float) failure
    with pytest.raises(ValueError, match="Label columns .* contain values that cannot be converted to numeric"):
        load_data(temp_files["csv_bad_label"], input_cols_str='f1,f2', label_cols_str='label')

def test_load_data_bad_feature_type(temp_files):
     """Test error when feature column contains non-numeric data."""
     with pytest.raises(ValueError, match="Input features contain non-numeric values"):
         load_data(temp_files["csv_bad_feat"], input_cols_str='f1,f2', label_cols_str='label')
