import numpy as np
from sklearn.preprocessing import MinMaxScaler


def minmax_scale(X):
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-9)
    X_scaled = X_std  # * (1 - 0) + 0  # * (max - min) + min
    return X_scaled


X = np.random.randint(-50, 100, size=(1, 1))
scaler = MinMaxScaler()
scaler.fit(X)

print(X, minmax_scale(X), scaler.transform(X), sep='\n')
