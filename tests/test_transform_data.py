import adadmire.main as am
import numpy as np


def test_transform_data():
    X = np.array([[1, 2, 3],
                  [6, 5, 4],
                  [7, 8, 9]])
    output = am.transform_data(X)
    expect = np.array([[0/6, 0/6, 0/6],
                       [5/6, 3/6, 1/6],
                       [6/6, 6/6, 6/6]])
    assert np.allclose(output, expect)
