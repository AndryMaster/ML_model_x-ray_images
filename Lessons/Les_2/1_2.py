import numpy as np


def calculate(products, cook):
    a = np.ceil(products.transpose().dot(cook)).astype(np.int32)
    print(f"Молоко, литры: {a[0]}",
          f"Яйца, штуки: {a[1]}",
          f"Мука, кг: {a[2]}",
          sep='\n')


products = np.array([
    [0.1, 2, 0.05],
    [0.2, 1, 0.2],
    [0.5, 3, 0.3]])

cook = [10, 32, 8]

calculate(products, cook)
