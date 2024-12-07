import pandas as pd

def load_data(file):
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

    # Split into features (X) and labels (Y)
    X = data.iloc[:, :-1].values  # All columns except the last are features
    Y = data.iloc[:, -1].values   # The last column is the label
    return X, Y

