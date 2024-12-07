# **Iris Dataset Example**

The **Iris dataset** contains measurements of flowers from three species of Iris:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

#### **Dataset Details**

- **Features (inputs):**
    - `sepal length` (in cm)
    - `sepal width` (in cm)
    - `petal length` (in cm)
    - `petal width` (in cm)
- **Label (output):**
    - Flower species:
        - `0`: Iris-setosa
        - `1`: Iris-versicolor
        - `2`: Iris-virginica

### **Step 1: Prepare the Dataset**

We’ll use the Iris dataset from the `scikit-learn` library, which provides it preformatted. First, save the dataset as CSV files for **training** and **testing**.

#### **Create Training Data**

Save the following Python script as `prepare_iris_data.py`:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Labels: 0 (setosa), 1 (versicolor), 2 (virginica)

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save training data
train_data = pd.DataFrame(X_train, columns=iris.feature_names)
train_data["label"] = y_train  # Add the labels as the last column
train_data.to_csv("iris_train.csv", index=False)

# Save test data (features only, no labels for inference)
test_data = pd.DataFrame(X_test, columns=iris.feature_names)
test_data.to_csv("iris_test.csv", index=False)

# Save test labels separately for evaluation
test_labels = pd.DataFrame(y_test, columns=["label"])
test_labels.to_csv("iris_test_labels.csv", index=False)

print("Training and test datasets saved as 'iris_train.csv' and 'iris_test.csv'.")
```

Run the script to generate:

1. `iris_train.csv` (training data with labels).
2. `iris_test.csv` (test data without labels, for predictions).
3. `iris_test_labels.csv` (test labels, for evaluation).

---

### **Step 2: Create a Neural Network**

We’ll create a neural network to classify the Iris dataset into three species.

#### **Command**

```bash
sygnals-nn create \
  --layers 4,10,3 \
  --activation relu,softmax \
  --loss categorical_crossentropy \
  --optimizer adam \
  --output iris_model.keras
```

#### **Explanation**

- **`--layers 4,10,3`**:
    - 4 input neurons (one for each feature: sepal length, sepal width, petal length, petal width).
    - 1 hidden layer with 10 neurons (using ReLU activation).
    - 3 output neurons (one for each flower species: setosa, versicolor, virginica).
- **`--activation relu,softmax`**:
    - ReLU activation for the hidden layer.
    - Softmax activation for the output layer to produce probabilities for multi-class classification.
- **`--loss categorical_crossentropy`**:
    - Used for multi-class classification tasks.
- **`--optimizer adam`**:
    - Adam optimizer for efficient training.
- **`--output iris_model.keras`**:
    - Saves the model to `iris_model.keras`.

#### **Output**

```
Model saved to iris_model.keras
```

---

### **Step 3: Train the Neural Network**

Now, train the model using the training data (`iris_train.csv`).

#### **Command**

```bash
sygnals-nn train \
  --model iris_model.keras \
  --data iris_train.csv \
  --epochs 200 \
  --batch-size 16 \
  --learning-rate 0.01
```

#### **Explanation**

- **`--model iris_model.keras`**: Load the model created in Step 2.
- **`--data iris_train.csv`**: Use the Iris training dataset.
- **`--epochs 200`**: Train the model for 200 iterations over the dataset.
- **`--batch-size 16`**: Process 16 samples at a time during training.
- **`--learning-rate 0.01`**: Set the speed of weight updates.

#### **Output**

```
Epoch 1/200
1/1 [==============================] - 0s 20ms/step - loss: 1.158 - accuracy: 0.333
...
Epoch 200/200
1/1 [==============================] - 0s 2ms/step - loss: 0.021 - accuracy: 1.000
Trained model saved to iris_model.keras
```

---

### **Step 4: Run Inference**

Use the trained model to predict the species of the flowers in the test data (`iris_test.csv`).

#### **Command**

```bash
sygnals-nn run \
  --model iris_model.keras \
  --input iris_test.csv \
  --output iris_predictions.csv
```

#### **Explanation**

- **`--model iris_model.keras`**: Load the trained model.
- **`--input iris_test.csv`**: Use the test data (features only, no labels).
- **`--output iris_predictions.csv`**: Save the predictions to a CSV file.

#### **Output**

```
Predictions saved to iris_predictions.csv
```

---

### **Step 5: Evaluate the Results**

#### **Inspect Predictions**

Open `iris_predictions.csv`. It contains the probabilities for each class (setosa, versicolor, virginica). For example:

```csv
0.99,0.01,0.00
0.05,0.85,0.10
0.00,0.20,0.80
```

- Row 1: 99% probability for setosa.
- Row 2: 85% probability for versicolor.
- Row 3: 80% probability for virginica.

#### **Compare with True Labels**

Compare the predictions in `iris_predictions.csv` with the true labels in `iris_test_labels.csv`. Use Python or a spreadsheet tool to calculate accuracy.

Here’s a simple Python script to calculate accuracy:

```python
import pandas as pd
import numpy as np

# Load predictions and true labels
predictions = pd.read_csv("iris_predictions.csv", header=None)
true_labels = pd.read_csv("iris_test_labels.csv")

# Get the predicted class (highest probability)
predicted_classes = np.argmax(predictions.values, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_classes == true_labels.values.ravel())
print(f"Accuracy: {accuracy * 100:.2f}%")
```

#### **Expected Output**

The trained neural network should achieve around **97-100% accuracy** on the Iris dataset.