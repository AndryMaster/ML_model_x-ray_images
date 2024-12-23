import sklearn
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

n_samples = 2500

x, y = datasets.make_moons(
    n_samples=n_samples,
    noise=0.2,
    random_state=5)

# print(x, y)
plt.scatter(x[:,0], x[:,1], c=y, s=20, marker='.')
plt.show()

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.3
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print(knn.get_params())
print(knn.score(x_test, y_test))
print(sklearn.metrics.accuracy_score(y_test, y_pred))


