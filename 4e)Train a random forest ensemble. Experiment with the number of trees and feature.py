
(Below command downlod the pkg aslo run and install jupyter

python -m pip install seaborn
python -m pip install notebook
python -m pip install pandas numpy seaborn matplotlib
jupyter notebook)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification



print("Do you have a CSV file to upload? (yes/no)")
response = input().lower().strip()

if response == "yes":
    filename = input("Enter the CSV file path: ")
    data = pd.read_csv(filename)

else:
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_classes=2,
        n_informative=6,
        n_redundant=2,
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




print("\nDataset Preview:")
print(data.head())




X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)




decision_tree = DecisionTreeClassifier(
    random_state=42
)

decision_tree.fit(X_train, y_train)

y_pred_tree = decision_tree.predict(X_test)

accuracy_tree = accuracy_score(y_test, y_pred_tree)

print("\nDecision Tree Accuracy:")
print(f"{accuracy_tree:.2f}")

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_tree))




random_forest = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    random_state=42,
    n_jobs=1
)

random_forest.fit(X_train, y_train)

y_pred_rf = random_forest.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)

print("\nRandom Forest Accuracy (100 trees):")
print(f"{accuracy_rf:.2f}")

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))




rf_experiment = RandomForestClassifier(
    n_estimators=50,
    max_features=3,
    random_state=42,
    n_jobs=1
)

rf_experiment.fit(X_train, y_train)

y_pred_rf_exp = rf_experiment.predict(X_test)

accuracy_rf_exp = accuracy_score(y_test, y_pred_rf_exp)

print("\nRandom Forest Accuracy (50 trees, max_features=3):")
print(f"{accuracy_rf_exp:.2f}")


print("\n========== MODEL COMPARISON ==========")

print(f"Decision Tree Accuracy: {accuracy_tree:.2f}")

print(f"Random Forest Accuracy (100 trees): {accuracy_rf:.2f}")

print(f"Random Forest Accuracy (50 trees): {accuracy_rf_exp:.2f}")




feature_importances = random_forest.feature_importances_

plt.figure(figsize=(10, 6))

plt.bar(
    range(len(feature_importances)),
    feature_importances,
    tick_label=data.columns[:-1],
    color='skyblue'
)

plt.title("Feature Importance in Random Forest")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



print("\nProgram Executed Successfully!")
