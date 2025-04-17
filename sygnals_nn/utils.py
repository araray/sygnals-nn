import pandas as pd
import numpy as np
import json # Added for JSON loading
import logging # Added for better feedback
import os # <<< Added missing import here

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _parse_col_indices_or_names(df_columns, cols_str):
    """
    Parses a comma-separated string of column indices or names
    into a list of valid column names for the given DataFrame columns.
    """
    if not cols_str:
        return []
    cols_list = []
    parts = cols_str.split(',')
    for part in parts:
        part = part.strip()
        try:
            # Try interpreting as an integer index
            idx = int(part)
            if 0 <= idx < len(df_columns):
                cols_list.append(df_columns[idx])
            else:
                raise ValueError(f"Column index {idx} is out of bounds.")
        except ValueError:
            # Interpret as a column name
            if part in df_columns:
                cols_list.append(part)
            else:
                raise ValueError(f"Column name '{part}' not found in data columns: {df_columns.tolist()}")
    return cols_list

def load_data(
    file_path: str,
    input_cols_str: str = None,
    label_cols_str: str = None,
    json_input_key: str = 'features',
    json_label_key: str = 'label',
    is_inference: bool = False,
    preprocessor=None, # Added: Optional preprocessor object
    text_col_for_preprocess: str = None # Added: Column to apply preprocessor to
    ) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Load data from CSV or JSON file, select features and labels based on
    column indices or names, and optionally apply a preprocessor.

    Args:
        file_path: Path to the dataset file (.csv or .json).
        input_cols_str: Comma-separated string of input column indices or names.
                        Required unless preprocessor handles feature extraction (e.g., TF-IDF).
        label_cols_str: Comma-separated string of label column indices or names.
                        Required if not is_inference.
        json_input_key: Key for input features in JSON objects.
        json_label_key: Key for label in JSON objects.
        is_inference: If True, load only inputs (X) and ignore labels.
        preprocessor: Optional fitted preprocessor object (e.g., scaler, vectorizer)
                      to apply to the data.
        text_col_for_preprocess: The specific column name to apply the text preprocessor to
                                 (e.g., for TF-IDF during inference).

    Returns:
        A tuple (X, Y):
            X: NumPy array of input features.
            Y: NumPy array of labels, or None if is_inference is True.

    Raises:
        ValueError: If file format is unsupported, columns are missing, or
                    configuration is invalid.
        FileNotFoundError: If the file_path does not exist.
    """
    logging.info(f"Loading data from: {file_path}")
    # Use the imported os module here
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # --- Load Data based on file type ---
    if file_path.lower().endswith(".csv"):
        try:
            # Try reading with header, then without if it fails
            try:
                data = pd.read_csv(file_path, header=0) # Assume header=0 is default
                logging.info(f"Loaded CSV with header: {data.columns.tolist()}")
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError): # Catch more potential errors
                 logging.warning(f"Could not parse CSV '{file_path}' with header, trying without header.")
                 # Reset file pointer if necessary (though usually not needed for read_csv)
                 data = pd.read_csv(file_path, header=None)
                 # Assign default numerical column names if no header
                 data.columns = [str(i) for i in range(data.shape[1])]
                 logging.info(f"Loaded CSV without header. Assigned columns: {data.columns.tolist()}")

        except Exception as e:
            raise ValueError(f"Error reading CSV file '{file_path}': {e}")

    elif file_path.lower().endswith(".json"):
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)

            # Handle common JSON structures:
            # 1. List of objects: [{'feat1': v, 'feat2': v, 'lbl': v}, ...]
            if isinstance(json_data, list) and json_data and all(isinstance(item, dict) for item in json_data):
                data = pd.DataFrame(json_data)
                logging.info(f"Loaded JSON as a list of objects. Columns: {data.columns.tolist()}")
            # 2. Dictionary of lists: {'feat1': [v,v,...], 'feat2': [v,v,...], 'lbl': [v,v,...]}
            elif isinstance(json_data, dict) and json_data and all(isinstance(val, list) for val in json_data.values()):
                 # Check if all lists have the same length
                list_lengths = [len(v) for v in json_data.values()]
                if len(set(list_lengths)) > 1: # Allow empty lists, but if multiple non-empty, lengths must match
                    non_empty_lengths = {l for l in list_lengths if l > 0}
                    if len(non_empty_lengths) > 1:
                        raise ValueError("JSON dictionary values (lists) must all have the same non-zero length.")
                data = pd.DataFrame(json_data)
                logging.info(f"Loaded JSON as a dictionary of lists. Columns: {data.columns.tolist()}")
            # Handle empty JSON file/list/dict gracefully
            elif not json_data:
                 logging.warning(f"JSON file '{file_path}' is empty.")
                 # Create an empty DataFrame with placeholder columns if needed downstream
                 # Or handle based on expected structure if possible
                 data = pd.DataFrame() # Or raise ValueError("JSON file is empty")

            else:
                # Try pandas direct read_json as a fallback
                try:
                    logging.warning("JSON structure not standard list-of-objects or dict-of-lists. Attempting pd.read_json.")
                    data = pd.read_json(file_path, orient='records') # Adjust 'orient' if needed
                    logging.info(f"Loaded JSON via pd.read_json. Columns: {data.columns.tolist()}")
                except Exception as e_pd:
                    raise ValueError(f"Unsupported JSON structure in '{file_path}'. Expected list of objects or dict of lists. pd.read_json error: {e_pd}")

        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON file '{file_path}': {e}")
        except Exception as e:
            raise ValueError(f"Error reading JSON file '{file_path}': {e}")

    else:
        # Potentially support raw text later if needed, but focus on CSV/JSON
        raise ValueError(f"Unsupported file format: {file_path}. Please use .csv or .json.")

    if data.empty:
        # Decide whether to raise error or return empty arrays
        logging.warning(f"Loaded data from '{file_path}' is empty.")
        # Returning empty arrays might be better than erroring immediately
        return np.array([]).astype(np.float32), np.array([]).astype(np.float32) if not is_inference else None
        # raise ValueError(f"Loaded data from '{file_path}' is empty.")


    # --- Select Columns ---
    X = None
    Y = None
    df_columns = data.columns

    # Handle case where preprocessor does the feature extraction (e.g., TF-IDF)
    if preprocessor and hasattr(preprocessor, 'transform') and text_col_for_preprocess:
        # Use _parse_col_indices_or_names to handle index/name for the text column
        parsed_text_cols = _parse_col_indices_or_names(df_columns, text_col_for_preprocess)
        if not parsed_text_cols:
             raise ValueError(f"Text column '{text_col_for_preprocess}' not found for preprocessing.")
        actual_text_col_name = parsed_text_cols[0] # Should only be one text column

        logging.info(f"Applying preprocessor to column: {actual_text_col_name}")
        # Important: preprocessor.transform expects an iterable (like a Series)
        # Ensure input is string and handle NaN
        text_series = data[actual_text_col_name].astype(str).fillna('')
        X = preprocessor.transform(text_series)
        # Convert sparse matrix to dense if needed by the model (common for TF-IDF)
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
        # Ensure selected columns exist before trying to access .values
        missing_cols = [col for col in input_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Input columns not found in data: {missing_cols}")

        X = data[input_cols].values
        # Apply preprocessor if provided and features were selected manually
        if preprocessor and hasattr(preprocessor, 'transform'):
             logging.info(f"Applying preprocessor to selected columns: {input_cols}")
             # Ensure data passed to transform is numeric if required by preprocessor (e.g., scaler)
             try:
                 numeric_X = data[input_cols].apply(pd.to_numeric, errors='coerce').fillna(0) # Coerce errors, fill NaN
                 X = preprocessor.transform(numeric_X)
             except Exception as e:
                  raise ValueError(f"Error applying preprocessor to columns {input_cols}: {e}. Ensure columns are numeric for scalers.")

             logging.info(f"Shape of features after preprocessing: {X.shape}")

    else:
        # Error if no input features are defined (unless handled by preprocessor above)
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
        # Ensure selected columns exist
        missing_cols = [col for col in label_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Label columns not found in data: {missing_cols}")

        Y = data[label_cols].values
        # Ensure Y is numeric
        try:
            # Attempt conversion, coercing errors to NaN first, then decide how to handle NaN
            Y_numeric = pd.DataFrame(Y, columns=label_cols).apply(pd.to_numeric, errors='coerce')
            if Y_numeric.isnull().any().any():
                 nan_cols = Y_numeric.columns[Y_numeric.isnull().any()].tolist()
                 logging.warning(f"Non-numeric or NaN values found in label columns: {nan_cols}. Check data or preprocessing steps.")
                 # Option 1: Raise error (safer)
                 raise ValueError(f"Label columns {nan_cols} contain non-numeric/NaN values.")
                 # Option 2: Fill NaN (e.g., with 0 or median, might hide issues)
                 # Y = Y_numeric.fillna(0).values.astype(np.float32)
            else:
                 Y = Y_numeric.values.astype(np.float32)

        except Exception as e: # Catch broader exceptions during conversion
            logging.error(f"Could not convert labels in columns {label_cols} to numeric: {e}. Check data.")
            raise ValueError(f"Label columns contain values that cannot be converted to numeric: {e}")
        logging.info(f"Shape of labels: {Y.shape}")


    # --- Final Checks and Type Conversion for X ---
    if X is None:
        # This case should ideally be caught earlier
        raise ValueError("Failed to extract input features (X). Check input column specifications and data format.")

    # Ensure X is a NumPy array if it came from pandas/sparse matrix
    if not isinstance(X, np.ndarray):
        try:
            X = np.array(X)
        except Exception as e:
            raise TypeError(f"Could not convert extracted features to NumPy array: {e}")

    # Ensure X is numeric if not already handled by a vectorizer
    # Check dtype AFTER potential preprocessing
    if X.size > 0 and not np.issubdtype(X.dtype, np.number):
        try:
            X = X.astype(np.float32)
        except ValueError as e:
            logging.error(f"Could not convert input features to numeric type (float32): {e}")
            # Try to identify non-numeric entries (might be slow for large X)
            try:
                # Efficiently check for non-numeric types if possible
                if isinstance(X.flat[0], str): # Check first element type as heuristic
                     problem_indices = np.where(~pd.to_numeric(X.ravel(), errors='coerce').notna())[0]
                     logging.error(f"Found non-numeric features at indices {problem_indices[:10]}: {X.flat[problem_indices[:10]]}...")
            except Exception as report_e:
                 logging.error(f"Could not report non-numeric features: {report_e}")
            raise ValueError(f"Input features contain non-numeric values that couldn't be converted: {e}")

    # Handle case where X might be empty after processing (e.g., empty input file)
    if X.size == 0:
         logging.warning("Input features (X) are empty after processing.")
         # Return empty array of appropriate shape (0 rows, unknown cols or 0 cols?)
         # This depends on how downstream code handles it. Let's return (0,0) shape.
         X = np.empty((0, 0), dtype=np.float32)


    logging.info(f"Final shape of features (X): {X.shape}")
    if Y is not None:
        logging.info(f"Final shape of labels (Y): {Y.shape}")

    return X, Y
