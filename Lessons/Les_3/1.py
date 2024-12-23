import numpy as np


def onehot_encoding(x):
    # print(arr, np.searchsorted(np.sort(np.unique(arr)), arr))
    return np.eye((U := np.unique(x)).size, dtype=int)[np.searchsorted(np.sort(U), x)]
    # res = np.zeros([arr.size, arr.max()], dtype=np.int32)
    # res[np.arange(arr.size), arr - 1] = 1
    # return res


arr = np.array([3, 2, 2, 1])
print(onehot_encoding(arr))
arr = np.array([1])
print(onehot_encoding(arr))
arr = np.array([0])
print(onehot_encoding(arr))
arr = np.array([111, 2, 46, 34])
print(onehot_encoding(arr))
arr = np.array([-1, -2])
print(onehot_encoding(arr))
