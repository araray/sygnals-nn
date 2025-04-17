import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def export_results(predictions_file, output_format, output_file):
    """
    Export predictions from a file to the specified format.

    Args:
        predictions_file (str): Path to the file containing predictions (usually CSV).
        output_format (str): The desired output format ('csv', 'json', 'raw').
        output_file (str): The path to save the exported results.
    """
    logging.info(f"Exporting predictions from {predictions_file} to {output_file} in format '{output_format}'")

    try:
        # Load predictions, assuming no header by default for prediction files
        predictions = pd.read_csv(predictions_file, header=None)
        logging.info(f"Loaded predictions data. Shape: {predictions.shape}")
    except FileNotFoundError:
        logging.error(f"Predictions file not found: {predictions_file}")
        raise
    except pd.errors.EmptyDataError:
        logging.warning(f"Predictions file is empty: {predictions_file}")
        # Create an empty file in the desired format or handle as appropriate
        open(output_file, 'w').close() # Create empty file
        logging.info(f"Empty predictions file exported to {output_file}")
        return
    except Exception as e:
        logging.error(f"Error loading predictions file {predictions_file}: {e}")
        raise

    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if output_format == "csv":
            # Save as CSV without index and header
            predictions.to_csv(output_file, index=False, header=False)
        elif output_format == "json":
            # Export as JSON list of lists (records orient gives list of dicts if headers exist)
            # Since we load with header=None, pandas assigns numeric headers.
            # 'values' orient might be closer to raw list of lists if needed.
            # Let's stick to 'records' for now, test should adapt.
            # If headers were ['0', '1', ...], orient='records' gives [{'0':v, '1':v}, ...]
            # If headers were None, read_csv gives default numeric headers.
            # Let's export as list of lists using 'values' orient for simplicity.
            predictions.to_json(output_file, orient="values", indent=2) # Use 'values' for list of lists
        elif output_format == "raw":
            # Treat 'raw' as simple CSV output without index or header
            predictions.to_csv(output_file, index=False, header=False)
        else:
            raise ValueError(f"Unsupported format: '{output_format}'. Choose from 'csv', 'json', or 'raw'.")

        logging.info(f"Predictions successfully exported to {output_file}")

    except Exception as e:
        logging.error(f"Error exporting predictions to {output_file}: {e}")
        raise
