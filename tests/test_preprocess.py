import pytest
import os
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sygnals_nn.preprocess import preprocess_data, _load_raw_data # Import helper too

# Fixture to create raw data files
@pytest.fixture
def raw_data_files(tmp_path):
    """Creates raw CSV and JSON files for preprocessing tests."""
    files = {}

    # CSV Data
    csv_path = tmp_path / "raw.csv"
    pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'text_data': ['great product', 'bad service', 'okay overall', 'great again', 'terrible'],
        'category': ['A', 'B', 'A', 'A', 'C'],
        'value1': [10.5, 11.2, 9.8, 10.1, 15.0],
        'value2': [100, 150, 120, 110, 190]
    }).to_csv(csv_path, index=False)
    files["csv"] = csv_path

    # JSON Data (List of Objects)
    json_path = tmp_path / "raw.json"
    json_data = [
        {'doc_id': 'j1', 'content': 'json is nice', 'class': 'Good', 'numeric': [5.1, 50]},
        {'doc_id': 'j2', 'content': 'json is bad', 'class': 'Bad', 'numeric': [6.2, 65]},
        {'doc_id': 'j3', 'content': 'json okay', 'class': 'Okay', 'numeric': [4.8, 55]},
    ]
    with open(json_path, 'w') as f: json.dump(json_data, f)
    files["json"] = json_path

    # Output paths
    files["out_data"] = tmp_path / "processed_data.csv"
    files["out_prep"] = tmp_path / "preprocessor.joblib"

    return files

# --- Test _load_raw_data Helper ---
def test_load_raw_csv(raw_data_files):
    """Test loading relevant columns from raw CSV."""
    data, text_col, label_col, feature_cols = _load_raw_data(
        raw_data_files["csv"],
        text_col='text_data',
        label_col='category',
        feature_cols=['value1', 'value2']
    )
    assert isinstance(data, pd.DataFrame)
    assert text_col == 'text_data'
    assert label_col == 'category'
    assert feature_cols == ['value1', 'value2']
    assert list(data.columns) == ['text_data', 'category', 'value1', 'value2'] # Only loaded specified

def test_load_raw_json(raw_data_files):
    """Test loading relevant keys from raw JSON."""
    data, text_col, label_col, feature_cols = _load_raw_data(
        raw_data_files["json"],
        text_col='content', # Use the name provided in the arg
        label_col='class_label', # Use the name provided in the arg
        feature_cols=['num1', 'num2'], # Use the names provided in the arg
        json_text_key='content', # Map to JSON key
        json_label_key='class', # Map to JSON key
        json_feature_key='numeric' # Map to JSON key
    )
    assert isinstance(data, pd.DataFrame)
    assert text_col == 'content'
    assert label_col == 'class_label'
    assert feature_cols == ['num1', 'num2']
    assert list(data.columns) == ['content', 'class_label', 'num1', 'num2']
    assert data['num1'][0] == 5.1 # Check values loaded correctly

# --- Test Preprocess Methods ---

def test_preprocess_tfidf(raw_data_files):
    """Test TF-IDF preprocessing method."""
    preprocess_data(
        input_path=str(raw_data_files["csv"]),
        output_data_path=str(raw_data_files["out_data"]),
        output_preprocessor_path=str(raw_data_files["out_prep"]),
        method='tfidf',
        text_col='text_data',
        tfidf_max_features=10 # Limit features for test
    )
    assert raw_data_files["out_data"].exists()
    assert raw_data_files["out_prep"].exists()

    # Load and check outputs
    processed_df = pd.read_csv(raw_data_files["out_data"])
    preprocessor = joblib.load(raw_data_files["out_prep"])

    assert isinstance(preprocessor, TfidfVectorizer)
    assert len(preprocessor.vocabulary_) <= 10
    assert processed_df.shape[0] == 5 # Number of input rows
    assert processed_df.shape[1] == len(preprocessor.vocabulary_) # Features should match vocab size
    # Check if original label column was added back (it should NOT be by default now)
    assert 'category' not in processed_df.columns

def test_preprocess_countvectorizer(raw_data_files):
    """Test CountVectorizer preprocessing method."""
    preprocess_data(
        input_path=str(raw_data_files["csv"]),
        output_data_path=str(raw_data_files["out_data"]),
        output_preprocessor_path=str(raw_data_files["out_prep"]),
        method='count',
        text_col='1' # Use index for text column
    )
    assert raw_data_files["out_data"].exists()
    assert raw_data_files["out_prep"].exists()
    preprocessor = joblib.load(raw_data_files["out_prep"])
    assert isinstance(preprocessor, CountVectorizer)

def test_preprocess_scaler(raw_data_files):
    """Test StandardScaler preprocessing method."""
    preprocess_data(
        input_path=str(raw_data_files["csv"]),
        output_data_path=str(raw_data_files["out_data"]),
        output_preprocessor_path=str(raw_data_files["out_prep"]),
        method='scale',
        feature_cols_str='value1,4' # Use name and index
    )
    assert raw_data_files["out_data"].exists()
    assert raw_data_files["out_prep"].exists()

    processed_df = pd.read_csv(raw_data_files["out_data"])
    preprocessor = joblib.load(raw_data_files["out_prep"])

    assert isinstance(preprocessor, StandardScaler)
    assert processed_df.shape == (5, 2) # 5 rows, 2 scaled columns
    # Check if data looks scaled (mean close to 0)
    np.testing.assert_almost_equal(processed_df.mean().values, [0.0, 0.0], decimal=5)
    np.testing.assert_almost_equal(processed_df.std().values, [1.0, 1.0], decimal=5)


def test_preprocess_labelencoder_csv(raw_data_files):
    """Test LabelEncoder preprocessing method on CSV."""
    preprocess_data(
        input_path=str(raw_data_files["csv"]),
        output_data_path=str(raw_data_files["out_data"]),
        output_preprocessor_path=str(raw_data_files["out_prep"]),
        method='label_encode',
        label_col='category' # Use name
    )
    assert raw_data_files["out_data"].exists()
    assert raw_data_files["out_prep"].exists()

    processed_df = pd.read_csv(raw_data_files["out_data"])
    preprocessor = joblib.load(raw_data_files["out_prep"])

    assert isinstance(preprocessor, LabelEncoder)
    assert processed_df.shape == (5, 1)
    assert processed_df.columns == ['category']
    # Check encoded values (A -> 0, B -> 1, C -> 2)
    expected_labels = [0, 1, 0, 0, 2]
    np.testing.assert_array_equal(processed_df['category'].values, expected_labels)
    assert list(preprocessor.classes_) == ['A', 'B', 'C']

def test_preprocess_labelencoder_json(raw_data_files):
    """Test LabelEncoder preprocessing method on JSON."""
    preprocess_data(
        input_path=str(raw_data_files["json"]),
        output_data_path=str(raw_data_files["out_data"]),
        output_preprocessor_path=str(raw_data_files["out_prep"]),
        method='label_encode',
        label_col='encoded_class', # Specify output column name
        json_label_key='class' # Specify input JSON key
    )
    assert raw_data_files["out_data"].exists()
    assert raw_data_files["out_prep"].exists()

    processed_df = pd.read_csv(raw_data_files["out_data"])
    preprocessor = joblib.load(raw_data_files["out_prep"])

    assert isinstance(preprocessor, LabelEncoder)
    assert processed_df.shape == (3, 1)
    assert processed_df.columns == ['encoded_class']
    # Check encoded values (Bad -> 0, Good -> 1, Okay -> 2)
    expected_labels = [1, 0, 2]
    np.testing.assert_array_equal(processed_df['encoded_class'].values, expected_labels)
    assert list(preprocessor.classes_) == ['Bad', 'Good', 'Okay']


# --- Test Error Handling ---
def test_preprocess_missing_column(raw_data_files):
    """Test error if specified column for method doesn't exist."""
    with pytest.raises(ValueError, match="Text column 'non_existent' not found"):
        preprocess_data(
            input_path=str(raw_data_files["csv"]),
            output_data_path=str(raw_data_files["out_data"]),
            output_preprocessor_path=str(raw_data_files["out_prep"]),
            method='tfidf',
            text_col='non_existent'
        )

def test_preprocess_method_requires_column(raw_data_files):
    """Test error if required column for a method is not provided."""
    with pytest.raises(ValueError, match="Text column.*must be specified for TF-IDF"):
        preprocess_data(
            input_path=str(raw_data_files["csv"]),
            output_data_path=str(raw_data_files["out_data"]),
            output_preprocessor_path=str(raw_data_files["out_prep"]),
            method='tfidf',
            text_col=None # Missing text_col
        )

def test_preprocess_unsupported_method(raw_data_files):
    """Test error for an unsupported preprocessing method."""
    with pytest.raises(ValueError, match="Unsupported preprocessing method: pca"):
        preprocess_data(
            input_path=str(raw_data_files["csv"]),
            output_data_path=str(raw_data_files["out_data"]),
            output_preprocessor_path=str(raw_data_files["out_prep"]),
            method='pca', # Unsupported
            feature_cols_str='value1,value2'
        )
