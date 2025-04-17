import pandas as pd
import numpy as np
import logging
import os
import json
import joblib # For saving/loading sklearn objects

# Import necessary sklearn components
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define supported methods
SUPPORTED_METHODS = ['tfidf', 'count', 'scale', 'label_encode']

def _load_raw_data(input_path, text_col=None, label_col=None, feature_cols=None, json_text_key=None, json_label_key=None, json_feature_key=None):
    """Loads raw data from CSV or JSON, identifying relevant columns/keys."""
    logging.info(f"Loading raw data from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input data file not found: {input_path}")

    # Convert input_path to string for consistent handling
    input_path_str = str(input_path)

    if input_path_str.lower().endswith(".csv"):
        try:
            # Try reading with header=0 first, then check if header looks like data
            data = pd.read_csv(input_path, header=0)
            # Heuristic: If all column names are purely numeric strings, assume no header
            if all(isinstance(col, str) and col.isdigit() for col in data.columns):
                logging.warning("CSV header looks like numeric data. Re-reading with header=None.")
                data = pd.read_csv(input_path, header=None)
                data.columns = [str(i) for i in range(data.shape[1])] # Assign default numeric string columns
                logging.info(f"Loaded CSV without header. Shape: {data.shape}, Assigned columns: {data.columns.tolist()}")
            else:
                logging.info(f"Loaded CSV with header. Shape: {data.shape}, Columns: {data.columns.tolist()}")

        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, IndexError) as e:
             logging.warning(f"Could not parse CSV '{input_path_str}' with header=0 (Error: {e}). Trying without header.")
             try:
                 data = pd.read_csv(input_path, header=None)
                 data.columns = [str(i) for i in range(data.shape[1])] # Assign default numeric string columns
                 logging.info(f"Loaded CSV without header. Shape: {data.shape}, Assigned columns: {data.columns.tolist()}")
             except Exception as e_nohead:
                 raise ValueError(f"Error reading CSV file '{input_path_str}' even without header: {e_nohead}") from e
        except Exception as e:
            raise ValueError(f"Error reading CSV file '{input_path_str}': {e}")


    elif input_path_str.lower().endswith(".json"):
        try:
            with open(input_path, 'r') as f:
                json_data = json.load(f)

            if not isinstance(json_data, list) or not all(isinstance(item, dict) for item in json_data):
                 # Allow dict of lists as well
                 if not (isinstance(json_data, dict) and json_data and all(isinstance(val, list) for val in json_data.values())):
                      raise ValueError("JSON input for preprocessing currently expects a list of objects or a dict of lists.")

            if isinstance(json_data, dict):
                 # Check lengths for dict of lists
                 list_lengths = [len(v) for v in json_data.values()]
                 if len(set(list_lengths)) > 1:
                     non_empty_lengths = {l for l in list_lengths if l > 0}
                     if len(non_empty_lengths) > 1:
                         raise ValueError("JSON dictionary values (lists) must all have the same non-zero length.")
                 data = pd.DataFrame(json_data)
                 logging.info(f"Loaded JSON (dict of lists). Shape: {data.shape}, Columns: {data.columns.tolist()}")
            elif not json_data: # Handle empty list
                 logging.warning(f"JSON file {input_path_str} is empty.")
                 return pd.DataFrame(), None, None, []
            else: # List of objects
                 data = pd.DataFrame(json_data)
                 logging.info(f"Loaded JSON (list of objects). Shape: {data.shape}, Columns: {data.columns.tolist()}")


            # Extract relevant data based on keys
            extracted = {}
            text_col_name = None
            label_col_name = None
            feature_col_names = []

            # Determine the actual key names from the first object for validation (if list of dicts)
            first_item_keys = data.columns # Use DataFrame columns now

            if text_col and json_text_key:
                 if json_text_key not in first_item_keys: raise ValueError(f"JSON text key '{json_text_key}' not found in JSON data columns: {first_item_keys.tolist()}.")
                 # Use provided text_col as the desired column name in the output DataFrame
                 text_col_name = text_col
                 extracted[text_col_name] = data[json_text_key] # Select series from DataFrame

            if label_col and json_label_key:
                 if json_label_key not in first_item_keys: raise ValueError(f"JSON label key '{json_label_key}' not found in JSON data columns: {first_item_keys.tolist()}.")
                 # Use provided label_col as the desired column name
                 label_col_name = label_col
                 extracted[label_col_name] = data[json_label_key] # Select series

            if feature_cols and json_feature_key:
                 if json_feature_key not in first_item_keys: raise ValueError(f"JSON feature key '{json_feature_key}' not found in JSON data columns: {first_item_keys.tolist()}.")
                 # Assume features under the key are lists or compatible structures
                 # Check if the column actually contains lists
                 if not data[json_feature_key].empty and isinstance(data[json_feature_key].iloc[0], list):
                     temp_features = data[json_feature_key].tolist() # Convert Series of lists to list of lists
                     num_features = len(temp_features[0]) if temp_features else 0
                     if not feature_cols or len(feature_cols) != num_features:
                         logging.warning(f"Number of feature names in --feature-cols ({len(feature_cols)}) doesn't match features found under key '{json_feature_key}' ({num_features}). Using default names.")
                         feature_col_names = [f"feature_{i}" for i in range(num_features)]
                     else:
                         feature_col_names = feature_cols # Use provided names
                     feature_df = pd.DataFrame(temp_features, columns=feature_col_names, index=data.index) # Ensure index aligns
                     for col in feature_col_names:
                         extracted[col] = feature_df[col] # Add features to extracted dict
                 else:
                     # If the feature key points to a simple column, treat it as a single feature
                     logging.warning(f"JSON feature key '{json_feature_key}' does not point to a list. Treating as single feature column.")
                     if len(feature_cols) != 1:
                         logging.warning(f"Provided {len(feature_cols)} feature names, but '{json_feature_key}' is a single column. Using first name: '{feature_cols[0]}'.")
                     feature_col_names = [feature_cols[0]] # Use only the first provided name
                     extracted[feature_col_names[0]] = data[json_feature_key]


            selected_data = pd.DataFrame(extracted) # Create DataFrame from selected data
            logging.info(f"Selected JSON data columns: {selected_data.columns.tolist()}")
            # Return the DataFrame and the derived/provided column names
            return selected_data, text_col_name, label_col_name, feature_col_names

        except Exception as e:
            raise ValueError(f"Error reading or processing raw JSON file '{input_path_str}': {e}")
    else:
        raise ValueError("Unsupported file type for raw data loading. Use .csv or .json.")


def preprocess_data(
    input_path: str | os.PathLike, # Accept Path objects
    output_data_path: str,
    output_preprocessor_path: str,
    method: str,
    text_col: str | None = None,
    label_col: str | None = None,
    feature_cols_str: str | None = None,
    json_text_key: str = 'text',
    json_label_key: str = 'label',
    json_feature_key: str = 'features',
    # Method specific args
    tfidf_max_features: int | None = None,
    # Add more args for other methods (e.g., scaler options)
    ):
    """
    Applies a specified preprocessing method to raw data and saves the
    transformed data and the fitted preprocessor object.

    Args:
        input_path: Path to the raw input data (CSV or JSON).
        output_data_path: Path to save the processed numerical data (CSV).
        output_preprocessor_path: Path to save the fitted preprocessor (joblib).
        method: The preprocessing method ('tfidf', 'count', 'scale', 'label_encode').
        text_col: Column name/index containing text (for tfidf, count).
        label_col: Column name/index containing labels (for label_encode).
        feature_cols_str: Comma-separated column names/indices for numerical features (for scale).
        json_text_key: Key for text data in JSON input.
        json_label_key: Key for label data in JSON input.
        json_feature_key: Key for numerical feature data in JSON input.
        tfidf_max_features: Max features for TF-IDF vectorizer.
    """
    logging.info(f"Starting preprocessing. Method: {method}")
    logging.info(f"Input: {input_path}, Output Data: {output_data_path}, Output Preprocessor: {output_preprocessor_path}")

    # --- Validate method FIRST ---
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported preprocessing method: '{method}'. Supported methods are: {SUPPORTED_METHODS}")

    # --- Validate required arguments based on method BEFORE loading ---
    if method in ['tfidf', 'count'] and not text_col:
        raise ValueError(f"Text column (--text-col) must be specified for method '{method}'.")
    if method == 'label_encode' and not label_col:
         raise ValueError(f"Label column (--label-col) must be specified for method '{method}'.")
    if method == 'scale' and not feature_cols_str:
         raise ValueError(f"Feature columns (--feature-cols) must be specified for method '{method}'.")

    feature_cols = feature_cols_str.split(',') if feature_cols_str else None

    # --- Load Raw Data ---
    # Pass the original text_col, label_col, feature_cols definitions
    # The helper will return the actual names found and the selected data
    # Note: _load_raw_data now returns only the *selected* columns based on the arguments.
    selected_data, text_col_name, label_col_name, feature_col_names = _load_raw_data(
        input_path, text_col, label_col, feature_cols,
        json_text_key, json_label_key, json_feature_key
    )

    # Check if loaded data is empty AFTER trying to select columns
    # If selected_data is empty, it means the specified columns weren't found or the file was empty.
    if selected_data.empty:
        # Check if the original file was actually empty or if column selection failed
        try:
            # Quick check: read first few bytes to see if file has content
            with open(input_path, 'rb') as f:
                has_content = bool(f.read(10))
            if not has_content:
                 logging.warning(f"Input file {input_path} appears to be empty. Skipping processing.")
            else:
                 logging.warning(f"Selected data for processing from {input_path} is empty (columns likely missing or incorrect). Skipping processing.")
        except Exception: # Handle potential errors reading the file again
             logging.warning(f"Could not verify content of {input_path}. Selected data is empty. Skipping processing.")

        # Create empty output files? Or just return? Let's create empty files.
        # Ensure directories exist first
        os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_preprocessor_path), exist_ok=True)
        open(output_data_path, 'w').close()
        # Cannot save an empty preprocessor, maybe skip or save None? Skip for now.
        logging.warning(f"Skipping saving preprocessor object as input data was empty or columns invalid.")
        return


    processed_data = None
    preprocessor = None

    # --- Apply Chosen Method ---
    # Now we use the validated column names (text_col_name, etc.) and the selected_data DataFrame
    if method == 'tfidf':
        # Validation already done, text_col_name should exist if text_col was provided
        if not text_col_name or text_col_name not in selected_data.columns:
            raise ValueError(f"Internal Error: Text column '{text_col_name}' not found in selected data.")
        logging.info(f"Applying TF-IDF to column: '{text_col_name}'")
        vectorizer = TfidfVectorizer(max_features=tfidf_max_features)
        # Ensure data is string and handle potential NaNs introduced during selection
        text_data_series = selected_data[text_col_name].astype(str).fillna('')
        processed_features = vectorizer.fit_transform(text_data_series)
        preprocessor = vectorizer
        feature_names = vectorizer.get_feature_names_out()
        processed_data = pd.DataFrame(processed_features.toarray(), columns=feature_names)
        logging.info(f"TF-IDF completed. Shape: {processed_data.shape}")

    elif method == 'count':
        if not text_col_name or text_col_name not in selected_data.columns:
            raise ValueError(f"Internal Error: Text column '{text_col_name}' not found in selected data.")
        logging.info(f"Applying Count Vectorizer to column: '{text_col_name}'")
        vectorizer = CountVectorizer()
        text_data_series = selected_data[text_col_name].astype(str).fillna('')
        processed_features = vectorizer.fit_transform(text_data_series)
        preprocessor = vectorizer
        feature_names = vectorizer.get_feature_names_out()
        processed_data = pd.DataFrame(processed_features.toarray(), columns=feature_names)
        logging.info(f"Count Vectorizer completed. Shape: {processed_data.shape}")

    elif method == 'scale':
        if not feature_col_names or not all(col in selected_data.columns for col in feature_col_names):
            missing = [col for col in feature_col_names if col not in selected_data.columns]
            raise ValueError(f"Internal Error: Feature columns {missing} not found in selected data.")
        logging.info(f"Applying StandardScaler to columns: {feature_col_names}")
        scaler = StandardScaler()
        try:
            # Select only the required columns for scaling
            numeric_data = selected_data[feature_col_names].apply(pd.to_numeric, errors='coerce')
            if numeric_data.isnull().any().any():
                 nan_cols = numeric_data.columns[numeric_data.isnull().any()].tolist()
                 logging.warning(f"NaN values found in feature columns {nan_cols} after converting to numeric. Filling with 0.")
                 numeric_data = numeric_data.fillna(0)
            processed_features = scaler.fit_transform(numeric_data)
            preprocessor = scaler
            processed_data = pd.DataFrame(processed_features, columns=feature_col_names)
            logging.info(f"Scaling completed. Shape: {processed_data.shape}")
        except Exception as e:
             raise ValueError(f"Error scaling features in columns {feature_col_names}: {e}")


    elif method == 'label_encode':
        if not label_col_name or label_col_name not in selected_data.columns:
            raise ValueError(f"Internal Error: Label column '{label_col_name}' not found in selected data.")
        logging.info(f"Applying LabelEncoder to column: '{label_col_name}'")
        encoder = LabelEncoder()
        # Handle potential NaNs introduced during selection
        labels = selected_data[label_col_name].fillna('__MISSING__')
        processed_labels = encoder.fit_transform(labels)
        preprocessor = encoder
        processed_data = pd.DataFrame({label_col_name: processed_labels})
        logging.info(f"Label Encoding completed. Classes: {encoder.classes_}")
        logging.info(f"Shape: {processed_data.shape}")


    # --- Save Processed Data and Preprocessor ---
    if processed_data is None or preprocessor is None:
        # This should ideally not be reached if method validation and application work
        raise RuntimeError(f"Preprocessing method '{method}' failed to produce output data or preprocessor object.")

    try:
        logging.info(f"Saving processed data to {output_data_path}...")
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
        processed_data.to_csv(output_data_path, index=False)
        logging.info("Processed data saved successfully.")
    except Exception as e:
        logging.error(f"Error saving processed data: {e}")
        raise

    try:
        logging.info(f"Saving preprocessor object to {output_preprocessor_path}...")
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_preprocessor_path), exist_ok=True)
        joblib.dump(preprocessor, output_preprocessor_path)
        logging.info("Preprocessor object saved successfully.")
    except Exception as e:
        logging.error(f"Error saving preprocessor object: {e}")
        raise
