# **Titanic Dataset Example**

### **Dataset Overview**

- **Source**: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)
    - Direct Download: [train.csv](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- **Goal**: Predict whether a passenger survived (`1`) or died (`0`).
- **Features** (input columns):
    - `Pclass`: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
    - `Sex`: Gender (`male` or `female`)
    - `Age`: Age of the passenger
    - `SibSp`: Number of siblings/spouses aboard
    - `Parch`: Number of parents/children aboard
    - `Fare`: Ticket fare
    - `Embarked`: Port of embarkation (`C`, `Q`, `S`)
- **Label** (output column):
    - `Survived`: `0` (died) or `1` (survived)

---

## **Step-by-Step Guide**

---

### **Step 1: Download the Dataset**

Download the dataset using the direct URL:

```bash
wget https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv -O titanic.csv
```

This will save the file as `titanic.csv`.

---

### **Step 2: Prepare the Dataset**

The Titanic dataset contains both numerical and categorical features. Neural networks work best with numerical data, so we’ll preprocess it:

1. **Convert categorical features** (e.g., `Sex`, `Embarked`) into numbers.
    - `Sex`: `male` → `1`, `female` → `0`.
    - `Embarked`: `C` → `0`, `Q` → `1`, `S` → `2`.
2. **Fill missing values** in `Age` and `Embarked` columns with the median.
3. **Split the data** into training and testing datasets.

#### **Preprocessing Script**

Save the following Python script as `prepare_titanic_data.py`:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Titanic dataset
df = pd.read_csv("titanic.csv")

# Select useful columns
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
label = "Survived"

# Preprocess the dataset
df["Sex"] = df["Sex"].map({"male": 1, "female": 0})  # Convert Sex to 1 (male) and 0 (female)
df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})  # Convert Embarked to numbers
df["Age"] = df["Age"].fillna(df["Age"].median())  # Fill missing ages with median
df["Embarked"] = df["Embarked"].fillna(2)  # Fill missing Embarked with the most common value
df["Fare"] = df["Fare"].fillna(df["Fare"].median())  # Fill missing Fare with median

# Extract features and label
X = df[features]
y = df[label]

# Standardize the features for better performance
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the training and test data
train_data = pd.DataFrame(X_train, columns=features)
train_data["Survived"] = y_train.values  # Add label column
train_data.to_csv("titanic_train.csv", index=False)

test_data = pd.DataFrame(X_test, columns=features)
test_data.to_csv("titanic_test.csv", index=False)

test_labels = pd.DataFrame(y_test, columns=["Survived"])
test_labels.to_csv("titanic_test_labels.csv", index=False)

print("Prepared training data saved to 'titanic_train.csv'")
print("Prepared test data saved to 'titanic_test.csv'")
print("Test labels saved to 'titanic_test_labels.csv'")
```

Run the script to generate:

1. `titanic_train.csv`: Training data (features + label).
2. `titanic_test.csv`: Test data (features only, no labels).
3. `titanic_test_labels.csv`: Test labels for evaluation.

---

### **Step 3: Create the Neural Network**

We’ll create a neural network to classify passengers as survived or not.

#### **Command**

```bash
sygnals-nn create \
  --layers 7,16,8,1 \
  --activation relu,relu,sigmoid \
  --loss binary_crossentropy \
  --optimizer adam \
  --output titanic_model.keras
```

#### **Explanation**

- **`--layers 7,16,8,1`**:
    - 7 input neurons (one for each feature: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`).
    - Hidden layer 1: 16 neurons (ReLU activation).
    - Hidden layer 2: 8 neurons (ReLU activation).
    - Output layer: 1 neuron (Sigmoid activation for binary classification).
- **`--activation relu,relu,sigmoid`**:
    - ReLU activation for hidden layers.
    - Sigmoid activation for the output layer.
- **`--loss binary_crossentropy`**:
    - Binary cross-entropy is used for binary classification problems.
- **`--optimizer adam`**:
    - Adam optimizer for efficient training.

#### **Output**

```
Model saved to titanic_model.keras
```

---

### **Step 4: Train the Neural Network**

Use the prepared training data (`titanic_train.csv`) to train the model.

#### **Command**

```bash
sygnals-nn train \
  --model titanic_model.keras \
  --data titanic_train.csv \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 0.001
```

#### **Explanation**

- **`--model titanic_model.keras`**: Load the model created in Step 3.
- **`--data titanic_train.csv`**: Use the Titanic training dataset.
- **`--epochs 100`**: Train for 100 iterations over the dataset.
- **`--batch-size 32`**: Process 32 samples at a time during training.
- **`--learning-rate 0.001`**: Set a small learning rate for stable convergence.

#### **Output**

```
Epoch 1/100
1/1 [==============================] - 0s 20ms/step - loss: 0.690 - accuracy: 0.500
...
Epoch 100/100
1/1 [==============================] - 0s 2ms/step - loss: 0.410 - accuracy: 0.850
Trained model saved to titanic_model.keras
```

---

### **Step 5: Run Inference**

Use the trained model to predict survival probabilities for passengers in the test set (`titanic_test.csv`).

#### **Command**

```bash
sygnals-nn run \
  --model titanic_model.keras \
  --input titanic_test.csv \
  --output titanic_predictions.csv
```

#### **Explanation**

- **`--model titanic_model.keras`**: Load the trained model.
- **`--input titanic_test.csv`**: Use the test data (features only, no labels).
- **`--output titanic_predictions.csv`**: Save predictions to a CSV file.

#### **Output**

```
Predictions saved to titanic_predictions.csv
```

---

### **Step 6: Evaluate the Results**

#### **Inspect Predictions**

Open `titanic_predictions.csv`. It contains survival probabilities:

```csv
0.89
0.12
0.75
0.05
...
```

- `0.89`: High probability of survival.
- `0.12`: Low probability of survival.

#### **Compare with True Labels**

Compare predictions in `titanic_predictions.csv` with true labels in `titanic_test_labels.csv`. Use the following Python script to calculate accuracy:

```python
import pandas as pd
import numpy as np

# Load predictions and true labels
predictions = pd.read_csv("titanic_predictions.csv", header=None)
true_labels = pd.read_csv("titanic_test_labels.csv")

# Convert probabilities to binary predictions (1 if p > 0.5 else 0)
predicted_classes = (predictions.values > 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(predicted_classes == true_labels.values.ravel())
print(f"Accuracy: {accuracy * 100:.2f}%")
```

#### **Expected Output**

The trained model should achieve around **80-85% accuracy**.

---

### **Summary of Workflow**

1. **Download the Titanic dataset**:

    ```bash
    wget https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv -O titanic.csv
    ```

2. **Prepare the data**:

    ```bash
    python prepare_titanic_data.py
    ```

3. **Create the model**:

    ```bash
    sygnals-nn create \
      --layers 7,16,8,1 \
      --activation relu,relu,sigmoid \
      --loss binary_crossentropy \
      --optimizer adam \
      --output titanic_model.keras
    ```

4. **Train the model**:

    ```bash
    sygnals-nn train \
      --model titanic_model.keras \
      --data titanic_train.csv \
      --epochs 100 \
      --batch-size 32 \
      --learning-rate 0.001
    ```

5. **Run inference**:

    ```bash
    sygnals-nn run \
      --model titanic_model.keras \
      --input titanic_test.csv \
      --output titanic_predictions.csv
    ```

6. **Evaluate the results**:
    Use the provided Python script to calculate accuracy.

---

### **Summary**

1. **Select data for training/testing**: Choose a dataset with inputs (features) and outputs (labels).
2. **Transform the data**: Clean, normalize, and format the data into CSV files.
3. **Create the neural network**: Define the structure with `sygnals-nn create`.
4. **Train the neural network**: Teach the network using training data.
5. **Test the neural network**: Evaluate it with testing data.
6. **Infer with real data**: Use the trained model for predictions.