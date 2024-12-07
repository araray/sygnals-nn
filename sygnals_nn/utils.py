import pandas as pd

def load_data(file, inference=False):
    """
    Load data from CSV, JSON, or raw text file.
    """
    # Handle CSV files
    if file.endswith(".csv"):
        data = pd.read_csv(file)
    # Handle JSON files
    elif file.endswith(".json"):
        data = pd.read_json(file)
    # Handle raw text files
    else:
        data = pd.read_csv(file, delimiter=" ", header=None)

    if inference:
        # In inference mode, we assume all columns are features
        # return (X, None)
        # Ensure at least one column exists
        if data.shape[1] < 1:
            raise IndexError("No features found for inference.")
        return data.values, None  # Return two values: X and None for label

    # Training mode: last column is label
    if data.shape[1] < 2:
        raise IndexError("No label column found.")

    # Split into features (X) and labels (Y)
    X = data.iloc[:, :-1].values  # All columns except the last are features
    Y = data.iloc[:, -1].values   # The last column is the label
    return X, Y
