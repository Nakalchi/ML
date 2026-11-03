(Below command downlod the pkg aslo run and install jupyter

python -m pip install seaborn
python -m pip install notebook
python -m pip install pandas numpy seaborn matplotlib
jupyter notebook)
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification

from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# Generate Synthetic Dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    n_clusters_per_class=1,
    random_state=42
)

# Create DataFrame
df = pd.DataFrame(
    X,
    columns=[f'Feature_{i}' for i in range(1, 11)]
)

df['Target'] = y

print("First 5 Records:")
print(df.head())


# Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# k-Fold Cross Validation
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nk-Fold Cross Validation:")

for train_index, val_index in kf.split(X_train):
    print("TRAIN:", train_index)
    print("VALIDATION:", val_index)


# Stratified k-Fold Cross Validation
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nStratified k-Fold Cross Validation:")

for train_index, val_index in skf.split(X_train, y_train):
    print("TRAIN:", train_index)
    print("VALIDATION:", val_index)


# Train and Evaluate Using k-Fold
model = RandomForestClassifier(random_state=42)

accuracies = []

for train_index, val_index in kf.split(X_train):

    X_kf_train = X_train[train_index]
    X_kf_val = X_train[val_index]

    y_kf_train = y_train[train_index]
    y_kf_val = y_train[val_index]

    model.fit(X_kf_train, y_kf_train)

    y_pred = model.predict(X_kf_val)

    accuracy = accuracy_score(y_kf_val, y_pred)

    accuracies.append(accuracy)

print("\nAverage Accuracy from k-Fold: {:.2f}".format(np.mean(accuracies)))


# Hyperparameter Tuning Using GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    ),
    scoring='accuracy',
    n_jobs=1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(grid_search.best_score_)


# Final Model Evaluation
best_model = grid_search.best_estimator_

y_test_pred = best_model.predict(X_test)

print("\nTest Accuracy:")
print(accuracy_score(y_test, y_test_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Class 0', 'Class 1'],
    yticklabels=['Class 0', 'Class 1']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

