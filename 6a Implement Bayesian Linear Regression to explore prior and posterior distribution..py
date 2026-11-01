import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from google.colab import files
print("Do you have a CSV file to upload? (yes/no)")
response = input().lower()
if response == "yes":
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
else:
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 2 * X + 1 + np.random.randn(100, 1) * 2
    data = pd.DataFrame(
        np.hstack((X, y)),
        columns=["X", "y"]
    )
    filename = "synthetic_data.csv"
    data.to_csv(filename, index=False)
    print(f"Synthetic dataset saved as {filename}.")
data = pd.read_csv(filename)
print("Dataset Preview:")
print(data.head())
if response == "yes":
    # Encode object columns to numbers
    le = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = le.fit_transform(data[col].astype(str))
    # Dynamically select target (last column) and features (all other columns)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
else:
    X = data[["X"]].values
    y = data["y"].values



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
bayesian_regressor = BayesianRidge()
bayesian_regressor.fit(X_train, y_train)
y_pred = bayesian_regressor.predict(X_test)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].set_title("Prior Distribution (Assumed)")
ax[0].hist(
    np.random.normal(0, 1, 1000),
    bins=50,
    alpha=0.7,
    color='blue',
    label="Prior"
)
ax[0].legend()
ax[1].set_title("Posterior Distribution (After Fitting)")
ax[1].hist(
    bayesian_regressor.coef_,
    bins=50,
    alpha=0.7,
    color='green',
    label="Posterior"
)
ax[1].legend()
plt.show()
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
plt.figure(figsize=(8, 6))
# If X has multiple dimensions, plot using the first column for visualization
plot_x_test = X_test[:, 0] if X_test.shape[1] > 1 else X_test.ravel()
sorted_idx = np.argsort(plot_x_test)
plt.scatter(
    plot_x_test,
    y_test,
    color="blue",
    label="True values",
    alpha=0.7
)
plt.plot(
    plot_x_test[sorted_idx],
    y_pred[sorted_idx],
    color="red",
    label="Predicted values",
    linewidth=2
)








plt.title("Bayesian Linear Regression: True vs Predicted Values")
plt.xlabel("Feature 1" if response == "yes" else "X")
plt.ylabel("y")
plt.legend()
plt.show()
