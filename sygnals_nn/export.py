import pandas as pd

def export_results(predictions_file, output_format, output_file):
    """
    Export predictions to the specified format.
    """
    predictions = pd.read_csv(predictions_file, header=None)

    if output_format == "csv":
        predictions.to_csv(output_file, index=False, header=False)
    elif output_format == "json":
        predictions.to_json(output_file, orient="records")
    elif output_format == "raw":
        predictions.to_string(output_file, index=False, header=False)
    else:
        raise ValueError("Unsupported format. Choose from 'csv', 'json', or 'raw'.")

    print(f"Predictions exported to {output_file}")

