import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression

np.random.seed(42)

X, y = make_regression(
    n_samples=1000,
    n_features=10,
    noise=15,
    random_state=42
)

data = pd.DataFrame(
    X,
    columns=[f"X{i}" for i in range(1, 11)]
)

data["y"] = y

print(data.head())

X_train, X_test, y_train, y_test = train_test_split(
    data.drop("y", axis=1),
    data["y"],
    test_size=0.2,
    random_state=42
)

ridge = Ridge(alpha=1.0)

ridge.fit(X_train, y_train)

ridge_pred = ridge.predict(X_test)

ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
ridge_r2 = r2_score(y_test, ridge_pred)

print(f"Ridge RMSE: {ridge_rmse}")
print(f"Ridge R^2: {ridge_r2}")

lasso = Lasso(alpha=0.1)

lasso.fit(X_train, y_train)

lasso_pred = lasso.predict(X_test)

lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
lasso_r2 = r2_score(y_test, lasso_pred)

print(f"Lasso RMSE: {lasso_rmse}")
print(f"Lasso R^2: {lasso_r2}")

print("Lasso Coefficients:")
print(lasso.coef_)

elastic_net = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5
)

elastic_net.fit(X_train, y_train)

elastic_net_pred = elastic_net.predict(X_test)

elastic_net_rmse = np.sqrt(
    mean_squared_error(y_test, elastic_net_pred)
)

elastic_net_r2 = r2_score(
    y_test,
    elastic_net_pred
)

print(f"ElasticNet RMSE: {elastic_net_rmse}")
print(f"ElasticNet R^2: {elastic_net_r2}")

metrics = pd.DataFrame({
    "Model": ["Ridge", "Lasso", "ElasticNet"],
    "RMSE": [ridge_rmse, lasso_rmse, elastic_net_rmse],
    "R^2": [ridge_r2, lasso_r2, elastic_net_r2]
})

print(metrics)

sns.barplot(data=metrics, x="Model", y="RMSE")
plt.title("Model RMSE Comparison")
plt.show()
