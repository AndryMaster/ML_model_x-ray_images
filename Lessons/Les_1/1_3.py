import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

ds = pd.read_csv('penguins.csv')
ds = ds.dropna()
# print(ds)

labels = np.array(ds['species'].values)
labels = np.unique(labels, return_inverse=True)[1]  # .reshape((-1, 1))
features = np.array([ds['bill_length_mm'].values, ds['bill_depth_mm'].values], dtype=np.float32)
features = features.transpose()

# Print & Show
# print(labels.shape, features.shape)
# print(labels, features, labels.shape, features.shape, sep='\n')
# plt.scatter(features[:, 0], features[:, 1], s=15, c=np.unique(labels, return_inverse=True)[1])
# plt.show()

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=123)

# Learning
scores = []

for w_type in ['uniform', 'distance']:
    for i in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=i, weights=w_type)
        knn.fit(train_features, train_labels)
        scores.append(knn.score(test_features, test_labels))
        # print(f"Type: {w_type}  Neighbors: {i}  Score: {knn.score(test_features, test_labels)}")

print(f"Best accuracy: {max(scores) :.6f}")
print(f"Worst accuracy: {min(scores) :.6f}")
