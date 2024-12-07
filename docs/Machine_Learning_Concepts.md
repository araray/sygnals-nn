## **Machine Learning: key concepts**

Imagine you want to teach a computer how to recognize cats in pictures. To do this, you provide the computer with many example images and tell it which ones contain cats and which don’t. The computer learns from these examples and creates a "model" to recognize cats in new, unseen pictures. This process is called **machine learning**.

Before diving into Sygnals-NN, let’s clarify some foundational machine learning concepts to ensure everyone understands what’s happening at each step.

---

### **Neural Networks**

A **neural network** is a type of machine learning model that mimics the way the human brain processes information. It consists of **layers of neurons**, which work together to detect patterns in data and make predictions. Neural networks are trained using **datasets** that include examples of what they should learn to predict.

#### **Key Components of a Neural Network**

1. **Input Layer**:
    - The first layer, which receives the input data (e.g., features like age, income, and gender).
    - For example, if your data has three features (`x1`, `x2`, `x3`), the input layer has three neurons.

2. **Hidden Layers**:
    - Layers between the input and output layers.
    - They perform transformations using mathematical functions called **activation functions** (e.g., ReLU, Sigmoid).
    - They serve to find patterns in the data.

3. **Output Layer**:
    - The final layer, which makes the prediction. For example:
        - One neuron: Used for binary classification (e.g., predicting "yes" or "no"). For example,  predicts whether an image contains a cat (yes or no).
        - Multiple neurons: Used for multi-class classification (e.g., classifying a flower species: setosa, versicolor, virginica). Another example, predicts which of three animals is in the image (e.g., cat, dog, rabbit).

---

### **Data Labeling**

#### **1. What is Data Labeling?**

Data labeling is the process of assigning a **label** (or output value) to each row in your dataset. This is what the neural network learns to predict.

#### **2. Why is Data Labeling Important?**

The neural network needs labeled examples to learn patterns. For example:

- In the Titanic dataset:
    - Inputs (features): `Pclass`, `Sex`, `Age`, `Fare`
    - Label: `Survived` (0 = didn’t survive, 1 = survived)

#### **3. How to Label Data**

##### **Example: Binary Classification**

If you’re trying to predict whether something is `True` or `False`, your dataset might look like this:

| x1   | x2   | Label (y) |
| ---- | ---- | --------- |
| 0.2  | 0.8  | 1         |
| 0.4  | 0.3  | 0         |
| 0.7  | 0.6  | 1         |

- `x1` and `x2` are features (the input to the network).
- `y` (label) is what the network should predict (1 for True, 0 for False).

#### **4. Labeling Example #1**

| Passenger Class (Pclass) | Sex    | Age  | Fare  | Survived |
| ------------------------ | ------ | ---- | ----- | -------- |
| 3                        | Male   | 22.0 | 7.25  | 0        |
| 1                        | Female | 38.0 | 71.28 | 1        |

In this example:

- The first row is labeled as `0` (did not survive).
- The second row is labeled as `1` (survived).

#### **5. Labeling Example #2: Multi-Class Classification**

If you're trying to classify something into multiple categories (e.g., dog, cat, rabbit), the labels might look like this:

| x1   | x2   | x3   | Label (y) |
| ---- | ---- | ---- | --------- |
| 0.1  | 0.2  | 0.7  | 0         |
| 0.8  | 0.3  | 0.5  | 1         |
| 0.5  | 0.4  | 0.3  | 2         |

Here, `y` represents the category (0 for dog, 1 for cat, 2 for rabbit).

---

### **Training**

When training a neural network, there are some important terms to understand:

#### **1. Epochs**

- **What are Epochs?**
    An **epoch** is one complete pass through your entire dataset during training. If your dataset has 100 examples and you train for 10 epochs, the network will see each example 10 times.

-  **What Does It Do?**

    The network learns from the data every time it sees it. More epochs allow the network to learn more.

- **Why Does It Matter?**

    - Too few epochs: The model doesn’t learn enough (underfitting).
    - Too many epochs: The model learns too much, including noise (overfitting).

- **How to Set It?**
    Start with 100 epochs and increase as needed.

- If you set `--epochs 100`, the network will go through the dataset 100 times during training.

#### **2. Batch Size**

- **What is Batch Size?**

    Instead of feeding the entire dataset to the network at once, the data is split into **batches**. For example:

    - If your dataset has 100 examples and your batch size is 10, the network processes 10 examples at a time.

-  **What Does It Do?**

    Batch size affects how quickly the network learns:

    - **Small batches**: The network updates its weights more often, which can lead to better generalization but slower training.
    - **Large batches**: The network processes more data at once, which makes training faster but can reduce generalization.

-  **How to Choose Batch Size**

    - Small datasets: Use a smaller batch size (e.g., `--batch-size 4`).
    - Large datasets: Use a larger batch size (e.g., `--batch-size 32` or `64`).

#### **3. Learning Rate**

- **What is Learning Rate?**

    The **learning rate** determines how much the network adjusts its weights after each step of training. Think of it as how fast the network learns.

- **What Does It Do?**

    - **Small learning rate**: The network learns slowly but is more precise.

    - **Large learning rate**: The network learns faster but might overshoot the optimal solution.

- **How to Choose Learning Rate**
    - Start with a small learning rate (e.g., `0.01`).
    - Experiment with larger or smaller values depending on the results.

-  **Example**

    Set the learning rate with `--learning-rate 0.01`.

---

### **Activation Functions**

#### **What are Activation Functions?**

Activation functions are applied to neurons to add non-linearity. Without them, neural networks would only learn simple relationships.

#### **Common Activation Functions**

1. **ReLU (Rectified Linear Unit)**:
    - Outputs `max(0, x)`. Works well for most hidden layers.
    - Use: `relu`
2. **Sigmoid**:
    - Outputs values between `0` and `1`. Suitable for binary classification.
    - Use: `sigmoid`
3. **Tanh**:
    - Outputs values between `-1` and `1`. Good for hidden layers.
    - Use: `tanh`
4. **Softmax**:
    - Outputs probabilities that sum to `1`. Used for multi-class classification.
    - Use: `softmax`

---

### **Loss Functions**

#### **What is a Loss Function?**

The loss function measures how far off the network’s predictions are from the true labels. During training, the optimizer works to minimize this value.

#### **Common Loss Functions**

1. **Binary Cross-Entropy**:
    - Use for binary classification (e.g., predicting survival, cat vs. no cat).
    - Use: `--loss binary_crossentropy`.
2. **Categorical Cross-Entropy**:
    - Use for multi-class classification (e.g., classifying flowers).
    - Use: `--loss categorical_crossentropy`.
3. **Mean Squared Error**:
    - Used for regression tasks (predicting continuous values like house prices).
    - Use: `--loss mean_squared_error`.

---

### **Optimizers**

#### **What is an Optimizer?**

The optimizer determines how the network adjusts its weights during training. It’s like the "strategy" the network uses to learn.

#### **Common Optimizers**

1. **Adam (Adaptive Moment Estimation)**:
    - Adaptive optimizer that works well for most problems.
    - Combines the benefits of other optimizers.
    - Use: `--optimizer adam`.
2. **SGD (Stochastic Gradient Descent)**:
    - Updates weights based on random subsets of data.
    - Use for simpler models.
    - Use: `--optimizer sgd`.
3. **RMSProp**:
    - Works well for recurrent neural networks or tasks with noisy data.
    - Use: `--optimizer rmsprop`.

---

## **Working with your data**

dddd

### **Steps required**

1. **Understand Your Goal and Dataset**
    (What problem are you solving? What data is available?)
    - Identify the data you’ll use to train the neural network (inputs and labels).
    - Identify the data you’ll use to test the neural network (for evaluation).

2. **Prepare and Process Data**
    (Make your data ready for the app.)
    - Clean the data.
    - Format it for the app (CSV, JSON, or raw text).
    - Split it into training and testing datasets.

3. **Create a Neural Network**
    (Define the structure of the neural network using Sygnals-NN.)

4. **Train the Neural Network**
    (Use your prepared training data to teach the neural network.)

5. **Test the Neural Network**
    (Use your testing data to evaluate its accuracy and performance.)

6. **Infer with Real Data**
    (Use the trained model to make predictions on new data.)

---

### **Step 1: Understand Your Goal and Dataset**

#### **What is the Goal?**

The first question to ask yourself is:

- What are you trying to predict or classify?
    Examples:
    - **Predict:** Will a customer buy a product?
    - **Classify:** Is this an email spam or not spam?

#### **What is Your Dataset?**

- **Training Data**: The data you use to teach the neural network. It must include both **inputs** (features) and **outputs** (labels). Labels are the outcomes the network is trained to predict.
    Example:
    - Features: `age`, `income`, `number of purchases`
    - Label: `buy` (1 if the customer bought the product, 0 if they didn’t)
- **Testing Data**: The data used to check if the neural network is learning correctly. This is **separate** from the training data and is used for evaluation.

#### **Tools You Can Use to Inspect Data**

- Use **Excel** or **Google Sheets** to view and organize your dataset.
- Use **R** or **Python** for deeper analysis (optional).
- Use **grep**, **awk**, or **sed** to inspect text-based data (e.g., CSV files).

---

### **Step 2: Prepare and Process Data**

This step is **crucial** for machine learning. Neural networks only work with numerical data, so you must ensure that your data is clean, numerical, and formatted properly.

#### **What You Need to Do**

1. **Clean Your Data**:

    - Handle missing values. For example:
        - Replace missing numeric values with the average or median.
        - Replace missing categorical values with the most frequent category.
    - Remove unnecessary columns (like text IDs or unrelated metadata).

    **Example in Excel**:

    - Use "Find & Replace" to replace missing cells with appropriate values.
    - Remove columns like "Customer ID" if it’s not relevant.

    **Example in R**:

    ```r
    data$Age[is.na(data$Age)] <- mean(data$Age, na.rm = TRUE)
    ```

    **Example with sed**:

    ```bash
    sed 's/,,/,0,/g' data.csv > cleaned_data.csv
    ```

2. **Convert Categorical Data to Numbers**:
    Neural networks don’t understand categories like `male` or `female`. Convert these to numbers (e.g., `male=1`, `female=0`).

    **Example in Excel**:

    - Add a new column and use a formula like `=IF(A1="male", 1, 0)`.

    **Example in grep/awk**:

    ```bash
    awk -F, '{ if ($2 == "male") $2=1; else if ($2 == "female") $2=0; print }' OFS=, data.csv > processed_data.csv
    ```

3. **Standardize Data (Optional but Recommended)**:
    Scale all numeric values to a range of `0` to `1` or normalize them to have a mean of `0` and standard deviation of `1`.

    - This prevents features with larger values (e.g., income in thousands) from dominating the training process.

    **In Excel**:

    - Use a formula like `=(A1-MIN(A:A))/(MAX(A:A)-MIN(A:A))` to scale a column.

4. **Split Data into Training and Testing Sets**:

    - **Training Data**: 70-80% of your dataset.
    - **Testing Data**: 20-30% of your dataset.

    **In Excel**:

    - Sort your data and manually divide it into two separate files:
        - `training_data.csv`
        - `testing_data.csv`

    **In R**:

    ```r
    set.seed(42)
    train_index <- sample(seq_len(nrow(data)), size = 0.8 * nrow(data))
    train_data <- data[train_index, ]
    test_data <- data[-train_index, ]
    ```

---

### **Step 3: Create a Neural Network**

The `create` command defines the architecture of the neural network.

#### **Command Syntax**

```bash
sygnals-nn create \
  --layers <layers> \
  --activation <activations> \
  --loss <loss> \
  --optimizer <optimizer> \
  --output <output>
```

#### **Parameters**

- `--layers`: Number of neurons in each layer. Format: `input,hidden1,hidden2,...,output`.
    Example: `--layers 4,8,1` (4 input neurons, 8 hidden neurons, 1 output neuron).

- `--activation`: Activation function for each layer.
    Example: `--activation relu,sigmoid`.

- `--loss`: Loss function to minimize during training.
    Examples:
    - `binary_crossentropy` (binary classification)
    - `categorical_crossentropy` (multi-class classification)
    - `mean_squared_error` (regression tasks)

- `--optimizer`: Optimizer for training. Examples: `adam`, `sgd`, `rmsprop`.

- `--output`: File to save the model.

#### **Example**

If you have:

- 4 input features (`Pclass`, `Sex`, `Age`, `Fare`)
- A binary classification task (Survived: `1` or `0`)

```bash
sygnals-nn create \
  --layers 4,8,1 \
  --activation relu,sigmoid \
  --loss binary_crossentropy \
  --optimizer adam \
  --output model.keras
```

---

### **Step 4: Train the Neural Network**

Use your training data to train the neural network.

- The `train` command trains the model using the prepared dataset.

    #### **Command Syntax**

    ```bash
    sygnals-nn train \
      --model <model_file> \
      --data <training_data_file> \
      --epochs <epochs> \
      --batch-size <batch_size> \
      --learning-rate <learning_rate>
    ```

    #### **Parameters**

    - `--epochs`: Number of complete passes through the training dataset.

    - `--batch-size`: Number of samples processed at a time.

    - `--learning-rate`: Step size for weight updates.

        **Example**:

```bash
sygnals-nn train \
  --model model.keras \
  --data training_data.csv \
  --epochs 200 \
  --batch-size 16 \
  --learning-rate 0.01
```

---

### **Step 5: Test the Neural Network**

The `run` command generates predictions using the trained model.

#### **Command Syntax**

```bash
sygnals-nn run \
  --model <model_file> \
  --input <test_data_file> \
  --output <predictions_file>
```

#### **Example**

```bash
sygnals-nn run \
  --model model.keras \
  --input testing_data.csv \
  --output predictions.csv
```

The predictions file will contain one prediction per row.

### **Step 6: Infer with Real Data**

Once you’ve verified the model performs well, use it to make predictions on **new, real-world data**.

1. Format the data like your training/testing data (ensure the same columns are present).
2. Use the `run` command to infer.

---

## **Workflow Summary**

1. **Prepare the Data**

2. **Create the Model**

3. **Train the Model**

4. **Run Inference**

5. **Evaluate the Results**