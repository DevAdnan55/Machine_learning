import sklearn

from sklearn.datasets import load_iris
print(load_iris())
load_iris(return_X_y=True)
X, y = load_iris(return_X_y=True)
from sklearn.linear_model import LinearRegression
Model = LinearRegression()
Model.fit(X, y)
print(Model.predict(X))
from sklearn.neighbors import KNeighborsRegressor
Model_2 = KNeighborsRegressor()
Model_2.fit(X, y)