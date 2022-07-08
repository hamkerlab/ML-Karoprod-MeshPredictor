import numpy as np

def one_hot(data, values):
    "Utility to create a one-hot matrix."

    c = len(values)
    N = len(data)

    res = np.zeros((N, c))

    for i in range(N):
        val = data[i]
        idx = list(values).index(val)
        res[i, idx] = 1.0

    return res