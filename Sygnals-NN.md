# **Sygnals-NN**

**Sygnals-NN** is a friendly command-line tool for creating, training, using, and managing neural networks. It’s designed for people with basic familiarity with data preparation and command-line interfaces, enabling experimentation with machine learning without extensive Python coding.

Designed with flexibility in mind, it adheres to the Unix philosophy by enabling modular tasks like creating models, preprocessing data, training models, running inferences, converting formats, and exporting predictions.

Whether you're working with data in **CSV** or **JSON** formats, Sygnals-NN empowers you to preprocess data and run machine learning experiments with ease.

### **Features**

-   **Model Creation**: Define and save neural network architectures (Dense, CNNs like Conv1D/Conv2D) with customizable layers, activation functions, loss functions, and optimizers.
-   **Data Preprocessing**: Built-in commands to preprocess raw data (CSV/JSON) using methods like TF-IDF, Count Vectorization, Scaling (StandardScaler), and Label Encoding. Saves processed data and reusable preprocessor objects.
-   **Training**: Train Keras models on your dataset (CSV/JSON), adjusting parameters like epochs, batch size, and learning rate. Optionally export the trained model to ONNX format directly after training.
-   **Inference**: Run predictions using pre-trained Keras or ONNX models on new datasets (CSV/JSON). Automatically applies saved preprocessors if provided.
-   **Model Conversion**: Convert trained Keras models to the ONNX format for broader deployment options.
-   **Export**: Save predictions to files in various formats like CSV or JSON.
-   **Input Formats**: Supports CSV and JSON for datasets. Handles different JSON structures (list of objects, dictionary of lists).
-   **Output Formats**: Saves models in `.keras` and `.onnx` formats. Saves predictions and processed data in CSV or JSON.

---

## **Installation**

### **Prerequisites**

- Python 3.10 or higher
- A terminal or command-line interface
- Virtual environment recommended for isolating dependencies

### **Installation Steps**

1. Clone the repository:

    ```bash
    git clone [https://github.com/araray/sygnals-nn](https://github.com/araray/sygnals-nn) sygnals-nn
    cd sygnals-nn
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    # Ensure you have tf2onnx and scikit-learn installed now
    ```

3. Install the package in editable mode:

    ```bash
    pip install -e .
    ```

4. Verify installation:

    ```bash
    sygnals-nn --help
    ```

    You should see output similar to this:

    ```
    Usage: sygnals-nn [OPTIONS] COMMAND [ARGS]...

      Sygnals-NN: A CLI tool for creating, training, running, and managing
      neural networks, with support for various data formats and preprocessing.

    Options:
      --help  Show this message and exit.

    Commands:
      convert    Convert a trained Keras model to ONNX format.
      create     Create and save a neural network architecture (Keras model).
      export     Export predictions from a file to a specified format.
      preprocess Preprocess raw data (CSV or JSON) using specified methods.
      run        Run inference using a trained Keras or ONNX model.
      train      Train a Keras model using the specified dataset.
    ```

---

## **Workflow Example (with Preprocessing)**

Let's imagine we have raw text data for classification in `raw_reviews.csv` with columns `review_text` and `sentiment` (positive/negative).

**1. Preprocess Text Data (TF-IDF):**

```bash
sygnals-nn preprocess \
  --input-data raw_reviews.csv \
  --output-data processed_features.csv \
  --output-preprocessor tfidf_vectorizer.joblib \
  --method tfidf \
  --text-col review_text \
  --tfidf-max-features 5000 # Optional: limit features


This reads raw_reviews.csv.
Applies TF-IDF to the review_text column.
Saves the numerical TF-IDF features to processed_features.csv.
Saves the fitted TfidfVectorizer object to tfidf_vectorizer.joblib.
Note: The original sentiment column might need separate encoding (see Step 1b).
1b. Preprocess Labels (Label Encoding):
sygnals-nn preprocess \
  --input-data raw_reviews.csv \
  --output-data processed_labels.csv \
  --output-preprocessor label_encoder.joblib \
  --method label_encode \
  --label-col sentiment


Reads raw_reviews.csv.
Applies Label Encoding to the sentiment column (e.g., 'positive' -> 1, 'negative' -> 0).
Saves the numerical labels to processed_labels.csv.
Saves the fitted LabelEncoder object to label_encoder.joblib.
Self-Correction: The current preprocess command saves only the processed column. You would need to manually combine processed_features.csv and processed_labels.csv into a single training_data.csv for the train command, ensuring rows align. Alternatively, the preprocess.py script could be enhanced to handle merging. For now, let's assume manual merging or modification. Let's create training_data.csv containing TF-IDF features and the encoded label.
2. Create a Model:
sygnals-nn create \
  --layers 5000,128,1 \
  --activation relu,sigmoid \
  --loss binary_crossentropy \
  --optimizer adam \
  --output sentiment_model.keras


Creates a Dense network suitable for the 5000 TF-IDF features.
3. Train the Model:
# Assuming training_data.csv has 5000 feature columns (0-4999) and 1 label column (5000)
sygnals-nn train \
  --model sentiment_model.keras \
  --data training_data.csv \
  --input-cols "0-4999" # Specify feature columns (adjust if needed)
  --label-cols "5000"    # Specify label column (adjust if needed)
  --epochs 50 \
  --batch-size 64 \
  --export-onnx sentiment_model.onnx # Optional: export to ONNX


Trains the model using the preprocessed numerical data.
Note: The --input-cols and --label-cols need to correctly point to the columns in the combined training_data.csv. Using ranges like "0-4999" might require implementation or using explicit comma-separated lists.
4. Prepare New Data for Inference:
Assume new reviews are in new_reviews.csv (only review_text column).
5. Run Inference (using saved preprocessor):
sygnals-nn run \
  --model sentiment_model.onnx # Use Keras or ONNX model
  --input-data new_reviews.csv \
  --output predictions.csv \
  --input-cols "review_text" # Specify the *original* text column name
  --preprocessor-path tfidf_vectorizer.joblib # Apply the saved TF-IDF vectorizer


Loads the new reviews.
Loads the saved tfidf_vectorizer.joblib.
The run command uses the preprocessor on the review_text column before feeding data to the model.
Saves predictions to predictions.csv.
Detailed Command Usage
(Add detailed explanations for preprocess and convert commands, similar to existing commands)
preprocess
Preprocesses raw data using various methods.
sygnals-nn preprocess [OPTIONS]


Options:
--input-data PATH: Path to raw data (CSV/JSON). [required]
--output-data PATH: Path to save processed numerical data (CSV). [required]
--output-preprocessor PATH: Path to save the fitted preprocessor object. [required]
--method [tfidf|count|scale|label_encode]: Preprocessing method. [required]
--text-col TEXT: Column name/index for text data (for tfidf, count).
--label-col TEXT: Column name/index for labels (for label_encode).
--feature-cols TEXT: Comma-separated columns for numerical features (for scale).
--json-text-key TEXT: JSON key for text data. [default: text]
--json-label-key TEXT: JSON key for label data. [default: label]
--json-feature-key TEXT: JSON key for numerical features. [default: features]
--tfidf-max-features INTEGER: Max features for TF-IDF.
convert
Converts a Keras model to ONNX format.
sygnals-nn convert [OPTIONS]


Options:
--keras-model PATH: Path to the Keras model (.keras). [required]
--output-onnx PATH: Path to save the ONNX model (.onnx). [required]
--input-signature TEXT: Optional: Input signature string (e.g., `'[tf.TensorSpec(shape=(None, 784), dtype
