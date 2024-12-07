# **Sygnals-NN**

**Sygnals-NN** is a friendly command-line tool for creating, training, and using neural networks. It’s designed for people with no programming experience but with basic familiarity with data preparation using tools like Excel, R, or text-processing utilities like sed, awk, and grep. With Sygnals-NN, you can experiment with machine learning without having to write Python code, while still allowing full customization of your neural network.

Designed with flexibility in mind, it adheres to the Unix philosophy by enabling modular tasks like creating models, training them, running inferences, and exporting predictions. This tool eliminates the need to write complex Python code and instead enables users to define and experiment with neural networks entirely from the command line.

Whether you're working with data in **Excel**, **R**, or processing files using simple tools like **sed**, **awk**, or **grep**, Sygnals-NN empowers you to preprocess data and run machine learning experiments with ease.

### **Features**

- **Model Creation**: Easily define and save neural network architectures with customizable layers, activation functions, loss functions, and optimizers.
- **Training**: Train neural networks on your dataset, adjusting parameters like epochs, batch size, and learning rate.
- **Inference**: Run predictions using pre-trained models on new datasets.
- **Export**: Save predictions to files in various formats like CSV, JSON, or plain text.
- **Input Formats**: Supports CSV, JSON, and raw text formats for datasets.
- **Output Formats**: Saves models in `.keras` format and predictions in CSV or JSON.
- **No programming knowledge required** — only data preparation and CLI usage!

---

## **Installation**

### **Prerequisites**

- Python 3.10 or higher
- A terminal or command-line interface
- Virtual environment recommended for isolating dependencies

### **Installation Steps**

1. Clone the repository:

    ```bash
    git clone https://github.com/araray/sygnals-nn sygnals-nn
    cd sygnals-nn
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Install the package in editable mode:

    ```bash
    pip install -e .
    ```

4. Verify installation:

    ```bash
    sygnals-nn --help
    ```

    You should see the following output:

    ```
    Usage: sygnals-nn [OPTIONS] COMMAND [ARGS]...
    
      A CLI tool for creating, training, and running neural networks.
    
    Options:
      --help  Show this message and exit.
    
    Commands:
      create   Create a neural network.
      train    Train a neural network.
      run      Run inference using a trained neural network.
      export   Export predictions to a file.
    ```

