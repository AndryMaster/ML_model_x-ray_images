import numpy as np

# x = np.array([
#     [1, 2, 6, 3],
#     [1, 1, 4, 3],
#     [-3, 4, -3, -3],
# ])
# y = np.array([
#     [1, 5],
#     [1, 3],
#     [6, 0],
#     [-2, 4],
# ])
#
# z = x.dot(y)
# print(y)
# print(x)
# print(np.sum(z), z)
# print(res)
# print(f"res: {res :.2f}   {res}")

# x = np.random.random([100000, 100000])
x = np.zeros((10000, 10000), dtype=np.float80)
x[:] = np.random.randn(*x.shape)
print(x, x.T, sep='\n')
