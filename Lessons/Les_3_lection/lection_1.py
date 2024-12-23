import numpy as np

x = np.array([2, 2, 5, -3, 0, 3])
y = np.array([2, 1, 9, -10, -5, 3])

y_pred = 3 * x - 5

evc = np.sum(np.square(y - y_pred))
mse = np.mean(np.square(y - y_pred))
mae = np.mean(np.abs(y - y_pred))

print(f"{evc=}\n{mse=}\n{mae=}")
