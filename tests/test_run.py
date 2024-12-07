import pytest
import os
import pandas as pd
from sygnals_nn.run import run_inference
from sygnals_nn.create import create_network

def test_run_inference():
    # Create a small model
    model_file = "test_infer_model.keras"
    create_network("2,2,1", "relu,sigmoid", "binary_crossentropy", "adam", model_file)

    # Create input data for inference (no label)
    data = """f1,f2
0,0
0,1
1,0
"""
    input_file = "infer_input.csv"
    with open(input_file, 'w') as f:
        f.write(data)

    output_file = "predictions.csv"
    if os.path.exists(output_file):
        os.remove(output_file)

    # With updated code handling inference=True, run inference should work now
    run_inference(model_file, input_file, output_file)
    assert os.path.exists(output_file)

    preds = pd.read_csv(output_file, header=None)
    assert preds.shape[0] == 3  # 3 samples
