import numpy as np
from sygnals_nn.utils import load_data
import pandas as pd
import pytest
import tempfile

def test_load_data_csv():
    data = """f1,f2,label
0.1,0.2,1
0.3,0.4,0
"""
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w', delete=False) as f:
        f.write(data)
        tmp_filename = f.name

    X, Y = load_data(tmp_filename)
    assert np.allclose(X, [[0.1, 0.2],[0.3,0.4]])
    assert np.allclose(Y, [1,0])

def test_load_data_missing_label():
    # Only one column means no label column
    data = """f1
0.1
0.3
"""
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w', delete=False) as f:
        f.write(data)
        tmp_filename = f.name

    with pytest.raises(IndexError):
        X, Y = load_data(tmp_filename)
