(Below command downlod the pkg aslo run and install jupyter

python -m pip install seaborn
python -m pip install notebook
python -m pip install pandas numpy seaborn matplotlib
jupyter notebook)
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB

from google.colab import files

print("Do you have a CSV file to upload? (yes/no)")
response = input().lower()

if response == "yes":
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
else:
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_classes=2,
        random_state=42
    )

    data = pd.DataFrame(
        X,
        columns=[f"Feature_{i}" for i in range(X.shape[1])]
    )

    data['Target'] = y

    filename = "synthetic_naive_bayes_data.csv"

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

naive_bayes = GaussianNB()

naive_bayes.fit(X_train, y_train)

y_pred = naive_bayes.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Naive Bayes Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

test_sample = [X_test[0]]

predicted_class = naive_bayes.predict(test_sample)

print(f"Test Sample: {test_sample}")
print(f"Predicted Class: {predicted_class[0]}")
