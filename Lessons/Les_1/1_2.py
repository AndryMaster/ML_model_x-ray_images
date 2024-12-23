import numpy as np

y = np.array(list(map(float, input().split())))
y_pred = np.array(list(map(float, input().split())))

R2 = 1 - (np.mean(np.square(y - y_pred)) / np.mean(np.square(y - np.mean(y))))

print(f"R2: {R2 :.2f}")
