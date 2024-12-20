import pandas as pd

def load_data(file, num_inputs, num_outputs, inference=False):
    """
    Load data from CSV, JSON, or raw text file and split into inputs (X) and outputs (Y).

    Parameters:
        file (str): Path to the dataset file.
        num_inputs (int): Number of input features (columns).
        num_outputs (int): Number of output features (columns).
        inference (bool): Whether to load the file for inference (only inputs, no labels).

    Returns:
        tuple: (X, Y) where X is the input data and Y is the output data (or None for inference).
    """
    # Load the dataset
    if file.endswith(".csv"):
        data = pd.read_csv(file, header=None)  # Assume no headers in the file
    elif file.endswith(".json"):
        data = pd.read_json(file)
    else:  # Assume raw text file
        data = pd.read_csv(file, delimiter=" ", header=None)

    # Check the number of columns in the data
    total_columns = data.shape[1]
    expected_columns = num_inputs + (num_outputs if not inference else 0)
    if total_columns < expected_columns:
        raise ValueError(
            f"Dataset has {total_columns} columns, but at least {expected_columns} are required "
            f"(num_inputs={num_inputs}, num_outputs={num_outputs})."
        )

    # Handle inference mode
    if inference:
        if total_columns != num_inputs:
            raise ValueError(
                f"Inference dataset must have exactly {num_inputs} columns (found {total_columns})."
            )
        return data.values, None  # Return features only for inference

    # Handle training mode
    X = data.iloc[:, :num_inputs].values  # First `num_inputs` columns are features
    Y = data.iloc[:, num_inputs:num_inputs + num_outputs].values  # Next `num_outputs` columns are labels

    # Ensure data types are numeric
    X = X.astype("float32")
    Y = Y.astype("float32")

    return X, Y
