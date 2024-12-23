import numpy as np


def onehot_encoding(x):
    return np.eye((U := np.unique(x)).size, dtype=int)[np.searchsorted(np.sort(U), x)]


import timeit

data = np.random.randint(0, 10_000, size=100_000, dtype=np.int16)

seconds = timeit.timeit("onehot_encoding(data)", globals=globals(), number=10)
print(f"Timeit: {seconds/10 :.3f} seconds")
