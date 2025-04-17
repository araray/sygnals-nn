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

def _load_raw_data(input_path, text_col=None, label_col=None, feature_cols=None, json_text_key=None, json_label_key=None, json_feature_key=None):
    """Loads raw data from CSV or JSON, identifying relevant columns/keys."""
    logging.info(f"Loading raw data from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input data file not found: {input_path}")

    # Convert input_path to string for consistent handling
    input_path_str = str(input_path)

    if input_path_str.lower().endswith(".csv"):
        try:
            data = pd.read_csv(input_path) # Assume header exists for raw data
            logging.info(f"Loaded CSV. Columns: {data.columns.tolist()}")
            # Prepare column lists based on input names/indices
            cols_to_extract = []
            text_col_name = None
            label_col_name = None
            feature_col_names = []
            selected_data = pd.DataFrame() # Initialize empty DataFrame

            if text_col:
                try:
                    idx = int(text_col)
                    if 0 <= idx < len(data.columns):
                         text_col_name = data.columns[idx]
                    else:
                         raise ValueError(f"Text column index '{text_col}' out of bounds.")
                except ValueError:
                    text_col_name = text_col
                if text_col_name not in data.columns: raise ValueError(f"Text column name '{text_col_name}' not found.")
                cols_to_extract.append(text_col_name)
                selected_data[text_col_name] = data[text_col_name]

            if label_col:
                try:
                    idx = int(label_col)
                    if 0 <= idx < len(data.columns):
                         label_col_name = data.columns[idx]
                    else:
                         raise ValueError(f"Label column index '{label_col}' out of bounds.")
                except ValueError:
                    label_col_name = label_col
                if label_col_name not in data.columns: raise ValueError(f"Label column name '{label_col_name}' not found.")
                cols_to_extract.append(label_col_name)
                selected_data[label_col_name] = data[label_col_name]


            if feature_cols:
                current_feature_cols = [] # Store successfully identified feature columns
                for col in feature_cols:
                    col_name = None
                    try:
                        idx = int(col)
                        if 0 <= idx < len(data.columns):
                             col_name = data.columns[idx]
                        else:
                             raise ValueError(f"Feature column index '{col}' out of bounds.")
                    except ValueError:
                         if col in data.columns:
                             col_name = col
                         else:
                             raise ValueError(f"Feature column name '{col}' not found.")
                    if col_name:
                        current_feature_cols.append(col_name)
                        selected_data[col_name] = data[col_name] # Add to selected data

                feature_col_names = current_feature_cols # Update the list of names
                cols_to_extract.extend(feature_col_names)


            # Return only necessary columns/series DataFrame and names
            # If no columns specified for method, selected_data might be empty, but raw data isn't necessarily
            return selected_data, text_col_name, label_col_name, feature_col_names

        except Exception as e:
            raise ValueError(f"Error reading raw CSV file '{input_path_str}': {e}")

    elif input_path_str.lower().endswith(".json"):
        try:
            with open(input_path, 'r') as f:
                json_data = json.load(f)

            if not isinstance(json_data, list) or not all(isinstance(item, dict) for item in json_data):
                 raise ValueError("JSON input for preprocessing currently expects a list of objects.")

            if not json_data: # Handle empty list
                 logging.warning(f"JSON file {input_path_str} is empty.")
                 return pd.DataFrame(), None, None, []


            # Extract relevant data based on keys
            extracted = {}
            text_col_name = None
            label_col_name = None
            feature_col_names = []

            # Determine the actual key names from the first object for validation
            first_item_keys = json_data[0].keys()

            if text_col and json_text_key:
                 if json_text_key not in first_item_keys: raise ValueError(f"JSON text key '{json_text_key}' not found in first JSON object.")
                 # Use provided text_col as the desired column name in the output DataFrame
                 text_col_name = text_col
                 extracted[text_col_name] = [item.get(json_text_key, '') for item in json_data]

            if label_col and json_label_key:
                 if json_label_key not in first_item_keys: raise ValueError(f"JSON label key '{json_label_key}' not found in first JSON object.")
                 # Use provided label_col as the desired column name
                 label_col_name = label_col
                 extracted[label_col_name] = [item.get(json_label_key) for item in json_data]

            if feature_cols and json_feature_key:
                 if json_feature_key not in first_item_keys: raise ValueError(f"JSON feature key '{json_feature_key}' not found in first JSON object.")
                 # Assume features under the key are lists or compatible structures
                 temp_features = [item.get(json_feature_key, []) for item in json_data]
                 # Convert to DataFrame for consistency, using feature_cols as names
                 num_features = len(temp_features[0]) if temp_features else 0
                 if not feature_cols or len(feature_cols) != num_features:
                     logging.warning(f"Number of feature names in --feature-cols ({len(feature_cols)}) doesn't match features found under key '{json_feature_key}' ({num_features}). Using default names.")
                     feature_col_names = [f"feature_{i}" for i in range(num_features)]
                 else:
                     feature_col_names = feature_cols # Use provided names
                 feature_df = pd.DataFrame(temp_features, columns=feature_col_names)
                 for col in feature_col_names:
                     extracted[col] = feature_df[col] # Add features to extracted dict

            data = pd.DataFrame(extracted)
            logging.info(f"Loaded JSON. Extracted columns: {data.columns.tolist()}")
            # Return the DataFrame and the derived/provided column names
            return data, text_col_name, label_col_name, feature_col_names

        except Exception as e:
            raise ValueError(f"Error reading raw JSON file '{input_path_str}': {e}")
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
    raw_data, text_col_name, label_col_name, feature_col_names = _load_raw_data(
        input_path, text_col, label_col, feature_cols,
        json_text_key, json_label_key, json_feature_key
    )

    # Check if loaded data is empty AFTER trying to select columns
    if raw_data.empty:
        logging.warning(f"Selected data for processing from {input_path} is empty. Skipping processing.")
        # Create empty output files? Or just return? Let's create empty files.
        # Ensure directories exist first
        os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_preprocessor_path), exist_ok=True)
        open(output_data_path, 'w').close()
        # Cannot save an empty preprocessor, maybe skip or save None? Skip for now.
        logging.warning(f"Skipping saving preprocessor object as input data was empty.")
        return


    processed_data = None
    preprocessor = None

    # --- Apply Chosen Method ---
    # Now we use the validated column names (text_col_name, etc.)
    if method == 'tfidf':
        # Validation already done, text_col_name should exist if text_col was provided
        if not text_col_name: raise ValueError("Internal Error: Text column name not found after loading.")
        logging.info(f"Applying TF-IDF to column: '{text_col_name}'")
        vectorizer = TfidfVectorizer(max_features=tfidf_max_features)
        text_data = raw_data[text_col_name].astype(str).fillna('')
        processed_features = vectorizer.fit_transform(text_data)
        preprocessor = vectorizer
        feature_names = vectorizer.get_feature_names_out()
        processed_data = pd.DataFrame(processed_features.toarray(), columns=feature_names)
        logging.info(f"TF-IDF completed. Shape: {processed_data.shape}")

    elif method == 'count':
        if not text_col_name: raise ValueError("Internal Error: Text column name not found after loading.")
        logging.info(f"Applying Count Vectorizer to column: '{text_col_name}'")
        vectorizer = CountVectorizer()
        text_data = raw_data[text_col_name].astype(str).fillna('')
        processed_features = vectorizer.fit_transform(text_data)
        preprocessor = vectorizer
        feature_names = vectorizer.get_feature_names_out()
        processed_data = pd.DataFrame(processed_features.toarray(), columns=feature_names)
        logging.info(f"Count Vectorizer completed. Shape: {processed_data.shape}")

    elif method == 'scale':
        if not feature_col_names: raise ValueError("Internal Error: Feature column names not found after loading.")
        logging.info(f"Applying StandardScaler to columns: {feature_col_names}")
        scaler = StandardScaler()
        try:
            numeric_data = raw_data[feature_col_names].apply(pd.to_numeric, errors='coerce')
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
        if not label_col_name: raise ValueError("Internal Error: Label column name not found after loading.")
        logging.info(f"Applying LabelEncoder to column: '{label_col_name}'")
        encoder = LabelEncoder()
        labels = raw_data[label_col_name].fillna('__MISSING__')
        processed_labels = encoder.fit_transform(labels)
        preprocessor = encoder
        processed_data = pd.DataFrame({label_col_name: processed_labels})
        logging.info(f"Label Encoding completed. Classes: {encoder.classes_}")
        logging.info(f"Shape: {processed_data.shape}")


    else:
        # This case should not be reachable if validation is done above
        raise ValueError(f"Unsupported preprocessing method: {method}")


    # --- Save Processed Data and Preprocessor ---
    if processed_data is None or preprocessor is None:
        raise RuntimeError("Preprocessing failed to produce output data or preprocessor object.")

    try:
        logging.info(f"Saving processed data to {output_data_path}...")
        os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
        processed_data.to_csv(output_data_path, index=False)
        logging.info("Processed data saved successfully.")
    except Exception as e:
        logging.error(f"Error saving processed data: {e}")
        raise

    try:
        logging.info(f"Saving preprocessor object to {output_preprocessor_path}...")
        os.makedirs(os.path.dirname(output_preprocessor_path), exist_ok=True)
        joblib.dump(preprocessor, output_preprocessor_path)
        logging.info("Preprocessor object saved successfully.")
    except Exception as e:
        logging.error(f"Error saving preprocessor object: {e}")
        raise
