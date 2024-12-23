import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ds = pd.read_csv('penguins.csv')
ds = ds.dropna()
# print(ds)

features = np.array([ds['bill_length_mm'].values, ds['bill_depth_mm'].values], dtype=np.float32)
features = features.transpose()

# Print & Show
# plt.scatter(features[:, 0], features[:, 1], s=15, c=np.unique(labels, return_inverse=True)[1])
# plt.show()
# print(features, features.shape)

n_row = int(input())
count = int(input())

row = features[n_row]

dist = np.sqrt(np.sum(np.square(features - row), axis=1)).reshape(-1, 1)
i_dist = list((i, d) for i, d in enumerate(dist))
i_dist.sort(key=lambda elem: elem[1])

idxs = list(i_dist[i][0] for i in range(1, count + 1))

# print(idxs, i_dist)
print(*features[idxs], sep='\n')
