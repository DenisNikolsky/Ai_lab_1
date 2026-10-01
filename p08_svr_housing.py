import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle

data = fetch_california_housing()
X, y = shuffle(data.data, data.target, random_state=7)

num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test,  y_test  = X[num_training:], y[num_training:]

test_points = [
    X_test[0],
    X_test[1],
    X_test[2],
]

# Перебор параметров C и epsilon
for C, eps in [(1.0, 0.1), (10.0, 0.1), (1.0, 0.5)]:
    sv_regressor = SVR(kernel='linear', C=C, epsilon=eps)
    sv_regressor.fit(X_train, y_train)
    y_test_pred = sv_regressor.predict(X_test)

    mse = mean_squared_error(y_test, y_test_pred)
    evs = explained_variance_score(y_test, y_test_pred)

    print(f"\n###### C={C}, epsilon={eps} ######")
    print("Mean squared error =", round(mse, 2))
    print("Explained variance score =", round(evs, 2))

    for i, tp in enumerate(test_points, 1):
        pred = sv_regressor.predict([tp])[0]
        print(f"  Test point {i}: predicted = {pred:.3f}, true = {y_test[i-1]:.3f}")