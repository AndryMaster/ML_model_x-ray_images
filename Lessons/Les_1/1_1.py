import numpy as np

x = np.array(list(map(float, input().split())))
y = np.array(list(map(float, input().split())))

print(f"MSE: {np.mean(np.square(x - y)) :.2f}")
print(f"MAE: {np.mean(np.abs(x - y)) :.2f}")
print(f"RMSE: {np.sqrt(np.mean(np.square(x - y))) :.2f}")
