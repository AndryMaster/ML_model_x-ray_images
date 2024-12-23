import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

X = np.array([
    list(map(int, '1963')),
    list(map(int, '3196')),
    list(map(int, '6319')),
    list(map(int, '9631')),
    list(map(int, '1010')),
])
y = np.array([2, 1, 3, -1, 1])
# model = LinearRegression().fit(X, y)
# model = Ridge(alpha=1.0).fit(X, y)

XL = np.hstack([np.array([[1] for _ in range(5)]), X])
print(XL)  # [ 1.53 -0.32  0.1  -0.21  0.37]
W_ = np.linalg.inv(XL.T @ XL + 1**2 * np.eye(5)) @ XL.T @ y  # 1**2 * np.eye(5)
b, W = W_[0], W_[1:]

y_pred = X @ W + b
# y_pred = model.predict(X)

evc = np.sum(np.square(y - y_pred))
mse = np.mean(np.square(y - y_pred))
mae = np.mean(np.abs(y - y_pred))
print(f"{evc=}\n{mse=}\n{mae=}")

print(y, y_pred, sep='\n')
print(b, W)
print(np.round(W_, 2))
# print(model.intercept_, np.round(model.coef_, 2), model.coef_)
