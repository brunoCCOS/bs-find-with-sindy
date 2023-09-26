import pandas as pd
import numpy as np

def load_data(filename):
    df = pd.read_csv(filename)
    return df


def get_discrete_integral_matrix(t, center=True):
    """Generative the discrete integration matrix."""
    n = np.size(t)
    td = t[1] - t[0]
    A = np.tril(td * np.ones((n, n)), 0)
    if center:
        for i in range(n):
            A[i, i] = A[i, i] / 2
        A[:, 0] = A[:, 0] - A[0, 0]
    return (A)

