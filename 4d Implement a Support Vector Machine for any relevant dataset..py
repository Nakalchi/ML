(Below command downlod the pkg aslo run and install jupyter

python -m pip install seaborn
python -m pip install notebook
python -m pip install pandas numpy seaborn matplotlib
jupyter notebook)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from google.colab import files

print("Do you have a CSV file to upload? (yes/no)")
response = input().lower()

if response == "yes":
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
else:
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_classes=2,
        random_state=42
    )

    data = pd.DataFrame(
        X,
        columns=[f"Feature_{i}" for i in range(X.shape[1])]
    )

    data['Target'] = y

    filename = "synthetic_data.csv"

    data.to_csv(filename, index=False)

    print(f"Synthetic dataset saved as {filename}.")

data = pd.read_csv(filename)

print("Dataset Preview:")
print(data.head())

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    random_state=42
)

svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(
    n_samples=100,
    centers=2,
    random_state=42,
    cluster_std=1.5
)

svm_model.fit(X, y)

plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap='coolwarm',
    edgecolor='k'
)

xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
    np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
)

Z = svm_model.decision_function(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)

plt.contour(
    xx,
    yy,
    Z,
    levels=[-1, 0, 1],
    linestyles=['--', '-', '--'],
    colors='k'
)

plt.title("SVM Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()
