import numpy as np


def snake(m: int, n: int) -> np.ndarray:
    a = np.arange(1, m * n + 1).reshape([m, n])
    a[1::2, :] = a[1::2, ::-1]
    return a


print(snake(3, 4))
