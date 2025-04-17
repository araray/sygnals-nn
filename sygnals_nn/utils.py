# sygnals_nn/utils.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json # Added for JSON loading
import logging # Added for better feedback
import os # Added missing import here

# Fix: Import necessary classes from sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _parse_col_indices_or_names(df_columns, cols_str):
    """
    Parses a comma-separated string of column indices or names
    into a list of valid column names for the given DataFrame columns.
    Handles potential non-string column names (e.g., integers from header=None).
    """
    if not cols_str:
        return []
    cols_list = []
    parts = cols_str.split(',')
    df_columns_str = [str(c) for c in df_columns] # Work with string versions of columns

    for part in parts:
        part = part.strip()
        try:
            # Try interpreting as an integer index
            idx = int(part)
            if 0 <= idx < len(df_columns):
                cols_list.append(df_columns[idx]) # Return original column name/int
            else:
                # Raise IndexError specifically for out-of-bounds, caught below
                raise IndexError(f"Column index {idx} is out of bounds for columns: {df_columns.tolist()}")
        except (ValueError, IndexError) as e: # Catch both int conversion errors and index errors
            # If it wasn't a valid index or int conversion failed, interpret as a column name (string comparison)
            if part in df_columns_str:
                # Find the original column name/int that matches the string part
                original_col = df_columns[df_columns_str.index(part)]
                cols_list.append(original_col)
            else:
                # Raise ValueError if it's neither a valid index nor a valid name
                # Include the original error type for clarity if it was IndexError
                if isinstance(e, IndexError):
                     raise ValueError(f"Column index {part} is out of bounds.") from e # Use original index error msg
                else:
                      raise ValueError(f"Column name '{part}' not found in data columns: {df_columns.tolist()}") from e
    return cols_list

def load_data(
    file_path: str | os.PathLike, # Accept Path objects
    input_cols_str: str | None = None,
    label_cols_str: str | None = None,
    json_input_key: str = 'features',
    json_label_key: str = 'label',
    is_inference: bool = False,
    preprocessor=None, # Added: Optional preprocessor object
    text_col_for_preprocess: str | None = None # Added: Column to apply preprocessor to
    ) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Load data from CSV or JSON file, select features and labels based on
    column indices or names, and optionally apply a preprocessor.

    Args:
        file_path: Path to the dataset file (.csv or .json). Can be str or Path object.
        input_cols_str: Comma-separated string of input column indices or names.
                        Required unless preprocessor handles feature extraction (e.g., TF-IDF).
        label_cols_str: Comma-separated string of label column indices or names.
                        Required if not is_inference.
        json_input_key: Key for input features in JSON objects.
        json_label_key: Key for label in JSON objects.
        is_inference: If True, load only inputs (X) and ignore labels.
        preprocessor: Optional fitted preprocessor object (e.g., scaler, vectorizer)
                      to apply to the data.
        text_col_for_preprocess: The specific column name/index to apply the text preprocessor to
                                 (e.g., for TF-IDF during inference).
    Returns:
        A tuple (X, Y):
            X: NumPy array of input features (float32).
            Y: NumPy array of labels (float32), or None if is_inference is True.
    Raises:
        ValueError: If file format is unsupported, columns are missing, or
                    configuration is invalid.
        FileNotFoundError: If the file_path does not exist.
    """
    logging.info(f"Loading data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Convert file_path to string for extension checking
    file_path_str = str(file_path)

    # --- Load Data based on file type ---
    if file_path_str.lower().endswith(".csv"):
        try:
            # Try reading WITHOUT header first
            data = pd.read_csv(file_path, header=None, skipinitialspace=True)
            # Assign default integer column names
            data.columns = list(range(data.shape[1]))
            logging.info(f"Attempted load CSV without header. Shape: {data.shape}, Assigned columns: {data.columns.tolist()}")
            # Basic check: If first row contains non-numeric convertible data, maybe there was a header
            first_row_numeric = pd.to_numeric(data.iloc[0], errors='coerce').notna().all()
            if not first_row_numeric:
                logging.warning("First row seems non-numeric after reading with header=None. Re-trying with header=0.")
                data = pd.read_csv(file_path, header=0, skipinitialspace=True)
                logging.info(f"Loaded CSV with header. Shape: {data.shape}, Columns: {data.columns.tolist()}")
            else:
                 logging.info("Loaded CSV assuming no header based on first row check.")

        except Exception as e:
            logging.warning(f"Initial CSV load failed or first row check indicated header ({e}). Trying with header=0.")
            try:
                 data = pd.read_csv(file_path, header=0, skipinitialspace=True)
                 logging.info(f"Loaded CSV with header. Shape: {data.shape}, Columns: {data.columns.tolist()}")
            except Exception as e_head:
                 raise ValueError(f"Error reading CSV file '{file_path_str}' with and without header: {e_head}") from e


    elif file_path_str.lower().endswith(".json"):
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)

            # Handle common JSON structures:
            if isinstance(json_data, list) and json_data and all(isinstance(item, dict) for item in json_data):
                data = pd.DataFrame(json_data)
                logging.info(f"Loaded JSON (list of objects). Shape: {data.shape}, Columns: {data.columns.tolist()}")
            elif isinstance(json_data, dict) and json_data and all(isinstance(val, list) for val in json_data.values()):
                list_lengths = [len(v) for v in json_data.values()]
                if len(set(list_lengths)) > 1:
                    non_empty_lengths = {l for l in list_lengths if l > 0}
                    if len(non_empty_lengths) > 1:
                        raise ValueError("JSON dictionary values (lists) must all have the same non-zero length.")
                data = pd.DataFrame(json_data)
                logging.info(f"Loaded JSON (dict of lists). Shape: {data.shape}, Columns: {data.columns.tolist()}")
            elif not json_data:
                 logging.warning(f"JSON file '{file_path_str}' is empty.")
                 data = pd.DataFrame()
            else:
                try:
                    logging.warning("JSON structure not standard. Attempting pd.read_json.")
                    data = pd.read_json(file_path, orient='records')
                    logging.info(f"Loaded JSON via pd.read_json. Shape: {data.shape}, Columns: {data.columns.tolist()}")
                except Exception as e_pd:
                    raise ValueError(f"Unsupported JSON structure in '{file_path_str}'. pd.read_json error: {e_pd}")

        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON file '{file_path_str}': {e}")
        except Exception as e:
            raise ValueError(f"Error reading JSON file '{file_path_str}': {e}")

    else:
        raise ValueError(f"Unsupported file format: {file_path_str}. Please use .csv or .json.")

    if data.empty:
        logging.warning(f"Loaded data from '{file_path_str}' is empty.")
        empty_X = np.array([]).reshape(0, 0).astype(np.float32)
        empty_Y = np.array([]).reshape(0, 0).astype(np.float32) if not is_inference else None
        return empty_X, empty_Y

    # --- Select Columns ---
    X = None
    Y = None
    df_columns = data.columns

    # Handle case where preprocessor does the feature extraction (e.g., TF-IDF)
    if preprocessor and hasattr(preprocessor, 'transform') and text_col_for_preprocess:
        parsed_text_cols = _parse_col_indices_or_names(df_columns, text_col_for_preprocess)
        if not parsed_text_cols:
             raise ValueError(f"Text column '{text_col_for_preprocess}' not found for preprocessing.")
        actual_text_col_name = parsed_text_cols[0]

        logging.info(f"Applying preprocessor to column: {actual_text_col_name}")
        text_series = data[actual_text_col_name].astype(str).fillna('')
        X = preprocessor.transform(text_series)
        if hasattr(X, "toarray"):
             logging.info("Converting sparse matrix from preprocessor to dense array.")
             X = X.toarray()
        logging.info(f"Shape of features after preprocessing: {X.shape}")

    # Handle case where features are selected from columns
    elif input_cols_str:
        input_cols = _parse_col_indices_or_names(df_columns, input_cols_str)
        if not input_cols:
            raise ValueError("No valid input columns selected.")
        logging.info(f"Selected input columns: {input_cols}")
        missing_cols = [col for col in input_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Input columns not found in data: {missing_cols}")

        X_selected = data[input_cols]

        # *** FIX for JSON list features ***
        # Check if X_selected is a single column containing lists (common for JSON features)
        if X_selected.shape[1] == 1 and not X_selected.empty and isinstance(X_selected.iloc[0, 0], list):
            logging.info(f"Detected list data in column '{input_cols[0]}'. Converting to NumPy array.")
            try:
                # Convert the list of lists into a 2D numpy array
                X = np.array(X_selected.iloc[:, 0].tolist())
                logging.info(f"Shape after converting list column: {X.shape}")
                # Now X is a NumPy array, proceed to preprocessor or type check
            except Exception as e:
                raise ValueError(f"Could not convert list data in column '{input_cols[0]}' to numeric array: {e}")
        else:
            # Not a column of lists, get values directly
            X = X_selected.values


        # Apply preprocessor if provided (AFTER potential list conversion)
        if preprocessor and hasattr(preprocessor, 'transform'):
             logging.info(f"Applying preprocessor to selected columns: {input_cols}")
             try:
                 # Ensure X is numeric before applying scaler-like preprocessors
                 if not np.issubdtype(X.dtype, np.number):
                     # Attempt conversion, raise error if it fails
                     try:
                         X_numeric = X.astype(float) # Try direct conversion first
                         # Check for NaNs introduced by conversion
                         if np.isnan(X_numeric).any():
                              raise ValueError("Non-numeric values found after attempting conversion.")
                         X = X_numeric
                     except (ValueError, TypeError) as e:
                         logging.error(f"Non-numeric values found in columns {input_cols} before applying preprocessor: {e}")
                         raise ValueError(f"Input features in columns {input_cols} contain non-numeric values.") from e

                 X = preprocessor.transform(X) # Apply transform
                 # Handle sparse output from scaler if necessary (less common)
                 if hasattr(X, "toarray"):
                     logging.info("Converting sparse matrix from preprocessor to dense array.")
                     X = X.toarray()
                 logging.info(f"Shape of features after preprocessing: {X.shape}")

             except Exception as e:
                  raise ValueError(f"Error applying preprocessor to columns {input_cols}: {e}.")
        # else: # X is already assigned from .values or list conversion


    else:
        # Neither preprocessor+text_col nor input_cols_str provided
        if not (preprocessor and text_col_for_preprocess):
            raise ValueError("Input columns must be specified via --input-cols, or a text preprocessor must be used with --text-col-for-preprocess.")


    # --- Select Labels (if not inference) ---
    if not is_inference:
        if not label_cols_str:
            raise ValueError("Label columns must be specified via --label-cols for training.")
        label_cols = _parse_col_indices_or_names(df_columns, label_cols_str)
        if not label_cols:
            raise ValueError("No valid label columns selected.")
        logging.info(f"Selected label columns: {label_cols}")
        missing_cols = [col for col in label_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Label columns not found in data: {missing_cols}")

        Y = data[label_cols].values
        # Ensure Y is numeric
        try:
            # Attempt direct conversion first
            Y_numeric = Y.astype(float)
            # Check for NaNs which indicate failed conversion
            if np.isnan(Y_numeric).any():
                 # Find original values that caused NaNs for better error message
                 original_labels_df = pd.DataFrame(Y, columns=label_cols)
                 nan_mask = pd.DataFrame(Y_numeric, columns=label_cols).isnull()
                 problematic_values = original_labels_df[nan_mask].apply(lambda row: [(col, row[col]) for col in nan_cols if pd.notna(row[col])], axis=1)
                 problematic_values = problematic_values[problematic_values.apply(len)>0] # Filter empty lists
                 logging.error(f"Found non-numeric/NaN values in labels: {problematic_values.to_string()}")
                 raise ValueError("Labels contain non-numeric or NaN values.")
            Y = Y_numeric # Keep as numpy array
            # Reshape if it's a single column
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)

        except (ValueError, TypeError) as e:
            logging.error(f"Could not convert labels in columns {label_cols} to numeric: {e}.")
            # Re-raise with a more specific message including the original error type
            raise ValueError(f"Label columns {label_cols} contain values that cannot be converted to numeric: {e}") from e
        logging.info(f"Shape of labels: {Y.shape}")


    # --- Final Checks and Type Conversion for X ---
    if X is None:
        # This should not happen if logic above is correct
        raise ValueError("Failed to extract input features (X).")

    # Ensure X is a NumPy array (might be sparse from vectorizer)
    if not isinstance(X, np.ndarray):
        try:
            # This case is less likely now after list conversion fix
            logging.warning(f"Converting X from type {type(X)} to NumPy array.")
            X = np.array(X)
        except Exception as e:
            raise TypeError(f"Could not convert extracted features to NumPy array: {e}")

    # Ensure X is numeric AFTER potential preprocessing (unless preprocessor was text vectorizer)
    # This check is important if X came directly from .values or list conversion without a scaler
    if X.size > 0 and not isinstance(preprocessor, (TfidfVectorizer, CountVectorizer)):
         if not np.issubdtype(X.dtype, np.number):
             try:
                 # Try converting, raising error if non-numeric present
                 X_numeric = X.astype(float)
                 if np.isnan(X_numeric).any():
                      raise ValueError("Input features contain non-numeric or NaN values after conversion attempt.")
                 X = X_numeric
             except (ValueError, TypeError) as e:
                 logging.error(f"Could not convert input features to numeric type: {e}")
                 raise ValueError(f"Input features contain non-numeric values that couldn't be converted: {e}") from e

    # --- Cast to float32 ---
    if X.size > 0:
        X = X.astype(np.float32)
    if Y is not None and Y.size > 0:
        # Check Y dtype again before casting, should be numeric now
        if not np.issubdtype(Y.dtype, np.number):
             # This should have been caught earlier
             raise TypeError(f"Labels (Y) are not numeric before final casting. Dtype: {Y.dtype}")
        Y = Y.astype(np.float32)


    logging.info(f"Final shape of features (X): {X.shape}, dtype: {X.dtype}")
    if Y is not None:
        logging.info(f"Final shape of labels (Y): {Y.shape}, dtype: {Y.dtype}")

    return X, Y
