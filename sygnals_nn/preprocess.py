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

    if input_path.lower().endswith(".csv"):
        try:
            data = pd.read_csv(input_path) # Assume header exists for raw data
            logging.info(f"Loaded CSV. Columns: {data.columns.tolist()}")
            # Prepare column lists based on input names/indices
            cols_to_extract = []
            text_col_name = None
            label_col_name = None
            feature_col_names = []

            if text_col:
                try:
                    idx = int(text_col)
                    text_col_name = data.columns[idx]
                except ValueError:
                    text_col_name = text_col
                if text_col_name not in data.columns: raise ValueError(f"Text column '{text_col_name}' not found.")
                cols_to_extract.append(text_col_name)

            if label_col:
                try:
                    idx = int(label_col)
                    label_col_name = data.columns[idx]
                except ValueError:
                    label_col_name = label_col
                if label_col_name not in data.columns: raise ValueError(f"Label column '{label_col_name}' not found.")
                cols_to_extract.append(label_col_name)

            if feature_cols:
                for col in feature_cols:
                    try:
                        idx = int(col)
                        feature_col_names.append(data.columns[idx])
                    except ValueError:
                         if col not in data.columns: raise ValueError(f"Feature column '{col}' not found.")
                         feature_col_names.append(col)
                cols_to_extract.extend(feature_col_names)

            # Return only necessary columns/series
            return data, text_col_name, label_col_name, feature_col_names

        except Exception as e:
            raise ValueError(f"Error reading raw CSV file '{input_path}': {e}")

    elif input_path.lower().endswith(".json"):
        try:
            with open(input_path, 'r') as f:
                json_data = json.load(f)

            if not isinstance(json_data, list) or not all(isinstance(item, dict) for item in json_data):
                 raise ValueError("JSON input for preprocessing currently expects a list of objects.")

            # Extract relevant data based on keys
            extracted = {}
            if text_col and json_text_key:
                 if json_text_key not in json_data[0]: raise ValueError(f"JSON text key '{json_text_key}' not found.")
                 extracted[text_col] = [item.get(json_text_key, '') for item in json_data] # Use provided text_col as name
                 text_col_name = text_col
            else: text_col_name = None

            if label_col and json_label_key:
                 if json_label_key not in json_data[0]: raise ValueError(f"JSON label key '{json_label_key}' not found.")
                 extracted[label_col] = [item.get(json_label_key) for item in json_data]
                 label_col_name = label_col
            else: label_col_name = None

            if feature_cols and json_feature_key:
                 if json_feature_key not in json_data[0]: raise ValueError(f"JSON feature key '{json_feature_key}' not found.")
                 # Assume features under the key are lists or compatible structures
                 temp_features = [item.get(json_feature_key, []) for item in json_data]
                 # Convert to DataFrame for consistency, assuming feature_cols are names for these
                 num_features = len(temp_features[0]) if temp_features else 0
                 if not feature_cols or len(feature_cols) != num_features:
                     logging.warning(f"Number of feature names in --feature-cols ({len(feature_cols)}) doesn't match features found under key '{json_feature_key}' ({num_features}). Using default names.")
                     feature_col_names = [f"feature_{i}" for i in range(num_features)]
                 else:
                     feature_col_names = feature_cols
                 feature_df = pd.DataFrame(temp_features, columns=feature_col_names)
                 for col in feature_col_names:
                     extracted[col] = feature_df[col]

            else: feature_col_names = []


            data = pd.DataFrame(extracted)
            logging.info(f"Loaded JSON. Extracted columns: {data.columns.tolist()}")
            return data, text_col_name, label_col_name, feature_col_names

        except Exception as e:
            raise ValueError(f"Error reading raw JSON file '{input_path}': {e}")
    else:
        raise ValueError("Unsupported file type for raw data loading. Use .csv or .json.")


def preprocess_data(
    input_path: str,
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

    feature_cols = feature_cols_str.split(',') if feature_cols_str else None

    # --- Load Raw Data ---
    raw_data, text_col_name, label_col_name, feature_col_names = _load_raw_data(
        input_path, text_col, label_col, feature_cols,
        json_text_key, json_label_key, json_feature_key
    )

    processed_data = None
    preprocessor = None

    # --- Apply Chosen Method ---
    if method == 'tfidf':
        if not text_col_name: raise ValueError("Text column (--text-col) must be specified for TF-IDF.")
        logging.info(f"Applying TF-IDF to column: '{text_col_name}'")
        vectorizer = TfidfVectorizer(max_features=tfidf_max_features)
        # Ensure text data is string and handle potential NaN/missing values
        text_data = raw_data[text_col_name].astype(str).fillna('')
        processed_features = vectorizer.fit_transform(text_data)
        preprocessor = vectorizer
        # Convert to DataFrame for saving (might be large)
        # Create meaningful column names
        feature_names = vectorizer.get_feature_names_out()
        processed_data = pd.DataFrame(processed_features.toarray(), columns=feature_names)
        logging.info(f"TF-IDF completed. Shape: {processed_data.shape}")

    elif method == 'count':
        if not text_col_name: raise ValueError("Text column (--text-col) must be specified for Count Vectorizer.")
        logging.info(f"Applying Count Vectorizer to column: '{text_col_name}'")
        vectorizer = CountVectorizer() # Add options like max_features if needed
        text_data = raw_data[text_col_name].astype(str).fillna('')
        processed_features = vectorizer.fit_transform(text_data)
        preprocessor = vectorizer
        feature_names = vectorizer.get_feature_names_out()
        processed_data = pd.DataFrame(processed_features.toarray(), columns=feature_names)
        logging.info(f"Count Vectorizer completed. Shape: {processed_data.shape}")

    elif method == 'scale':
        if not feature_col_names: raise ValueError("Feature columns (--feature-cols) must be specified for scaling.")
        logging.info(f"Applying StandardScaler to columns: {feature_col_names}")
        scaler = StandardScaler()
        # Ensure data is numeric, handle errors
        try:
            numeric_data = raw_data[feature_col_names].apply(pd.to_numeric, errors='coerce')
            if numeric_data.isnull().any().any():
                 logging.warning(f"NaN values found in feature columns {feature_col_names} after converting to numeric. Filling with 0.")
                 numeric_data = numeric_data.fillna(0) # Simple imputation, consider median/mean
            processed_features = scaler.fit_transform(numeric_data)
            preprocessor = scaler
            processed_data = pd.DataFrame(processed_features, columns=feature_col_names)
            logging.info(f"Scaling completed. Shape: {processed_data.shape}")
        except Exception as e:
             raise ValueError(f"Error scaling features in columns {feature_col_names}: {e}")


    elif method == 'label_encode':
        if not label_col_name: raise ValueError("Label column (--label-col) must be specified for label encoding.")
        logging.info(f"Applying LabelEncoder to column: '{label_col_name}'")
        encoder = LabelEncoder()
        # Handle potential missing values before encoding if necessary
        labels = raw_data[label_col_name].fillna('__MISSING__') # Replace NaN with a placeholder string
        processed_labels = encoder.fit_transform(labels)
        preprocessor = encoder
        # Output is just the encoded label column
        processed_data = pd.DataFrame({label_col_name: processed_labels})
        logging.info(f"Label Encoding completed. Classes: {encoder.classes_}")
        logging.info(f"Shape: {processed_data.shape}")


    else:
        raise ValueError(f"Unsupported preprocessing method: {method}")


    # --- Combine Processed Features with Unprocessed Columns (Optional) ---
    # If only specific columns were processed, you might want to merge back
    # other relevant columns (like IDs or unprocessed labels/features).
    # This example focuses on saving the *processed* part.
    # If vectorizing text, we might want to add the original label column back.
    if method in ['tfidf', 'count'] and label_col_name in raw_data.columns:
         logging.info(f"Adding original label column '{label_col_name}' to processed data.")
         # Ensure indices align if data was filtered/shuffled (shouldn't be here)
         processed_data[label_col_name] = raw_data[label_col_name].values


    # --- Save Processed Data and Preprocessor ---
    if processed_data is None or preprocessor is None:
        raise RuntimeError("Preprocessing failed to produce output data or preprocessor object.")

    try:
        logging.info(f"Saving processed data to {output_data_path}...")
        os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
        # Save processed data as CSV without index
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
