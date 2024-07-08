import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from pandas.plotting import scatter_matrix
import mglearn
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()
print(iris_data["target"])
print(iris_data["target_names"])
print(iris_data["feature_names"])
print(iris_data["data"])

x_train, x_test, y_train, y_test = train_test_split(
    iris_data["data"], iris_data["target"], random_state=0
)
print(x_test)
print(y_test)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

iris_data = pd.DataFrame(data=x_train, columns=iris.feature_names)

iris_data = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)


iris_target = pd.Series(iris_dataset.target, name="species")
grr = scatter_matrix(
    iris_data,
    c=iris_target,
    figsize=(15, 15),
    marker="o",
    hist_kwds={"bins": 20},
    s=60,
    alpha=0.8,
    cmap=mglearn.cm3,
)

plt.show()

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
x_new = np.array([[5, 2.9, 1, 0]])
print(x_new.shape)
pred = knn.predict(x_new)
print(pred)
print("Predicted target name: {}".format(iris_dataset["target_names"][pred]))
y_pred = knn.predict(x_test)
print(y_pred)
print("test score", (np.mean(y_pred == y_test)))
print("test score", knn.score(x_test, y_test))
