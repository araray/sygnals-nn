import pytest
import os
import pandas as pd
from sygnals_nn.export import export_results

def test_export_results_csv_to_csv():
    data = """0.8
0.2
0.9
"""
    pred_file = "predictions_input.csv"
    with open(pred_file, 'w') as f:
        f.write(data)
    output_file = "exported.csv"
    if os.path.exists(output_file):
        os.remove(output_file)

    export_results(pred_file, "csv", output_file)
    assert os.path.exists(output_file)

    with open(output_file, 'r') as f:
        lines = f.readlines()
    assert len(lines) == 3

def test_export_results_unsupported():
    with pytest.raises(ValueError):
        export_results("predictions_input.csv", "unsupported", "out.txt")
