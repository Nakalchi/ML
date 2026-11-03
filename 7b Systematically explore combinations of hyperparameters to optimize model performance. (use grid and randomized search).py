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
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from scipy.stats import randint

import matplotlib.pyplot as plt
import seaborn as sns

X, y = make_classification(
    n_samples=1000,
    n_features=12,
    n_informative=8,
    n_redundant=2,
    n_clusters_per_class=1,
    flip_y=0.03,
    random_state=42
)

df = pd.DataFrame(
    X,
    columns=[f'Feature_{i}' for i in range(1, 13)]
)

df['Target'] = y

print(df.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    random_state=42
)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    ),
    verbose=1,
    n_jobs=1
)

grid_search.fit(X_train, y_train)

print(
    "Best Parameters from Grid Search:",
    grid_search.best_params_
)

print(
    "Best Cross-Validation Accuracy from Grid Search:",
    grid_search.best_score_
)

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 15),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,
    scoring='accuracy',
    cv=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    ),
    verbose=1,
    n_jobs=1,
    random_state=42
)

random_search.fit(X_train, y_train)

print(
    "Best Parameters from Randomized Search:",
    random_search.best_params_
)

print(
    "Best Cross-Validation Accuracy from Randomized Search:",
    random_search.best_score_
)

best_model = random_search.best_estimator_

y_test_pred = best_model.predict(X_test)

print(
    "\nTest Accuracy:",
    accuracy_score(y_test, y_test_pred)
)

print(
    "\nClassification Report:\n",
    classification_report(y_test, y_test_pred)
)

conf_matrix = confusion_matrix(
    y_test,
    y_test_pred
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Class 0', 'Class 1'],
    yticklabels=['Class 0', 'Class 1']
)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

plt.show()
