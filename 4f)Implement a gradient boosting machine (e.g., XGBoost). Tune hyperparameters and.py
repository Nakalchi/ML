(Below command downlod the pkg aslo run and install jupyter

python -m pip install seaborn
python -m pip install notebook
python -m pip install pandas numpy seaborn matplotlib
jupyter notebook)




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification

from xgboost import XGBClassifier, plot_importance


# 2. Create or Upload a Dataset

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


# 3. Load and Display the Dataset

print("\nDataset Preview:")
print(data.head())


# 4. Preprocess the Data

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 5. Train a Basic XGBoost Model

xgb = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    n_jobs=1
)

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nXGBoost Accuracy (Default Parameters):")
print(f"{accuracy:.2f}")


# 6. Tune Hyperparameters Using GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        n_jobs=1
    ),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=1
)

grid_search.fit(X_train, y_train)


# Display Best Parameters

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(grid_search.best_score_)


# 7. Train the Final Model with Best Parameters

best_xgb = grid_search.best_estimator_

y_pred_best = best_xgb.predict(X_test)

accuracy_best = accuracy_score(y_test, y_pred_best)

print("\nXGBoost Accuracy (Tuned Parameters):")
print(f"{accuracy_best:.2f}")


# 8. Explore Feature Importance

plt.figure(figsize=(10, 6))

plot_importance(
    best_xgb,
    importance_type='weight',
    xlabel="Importance",
    ylabel="Features"
)

plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()


# 9. Evaluate the Model

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_best
    )
)


# 10. Compare Model Accuracy

print("\n========== MODEL COMPARISON ==========")

print(f"Default XGBoost Accuracy: {accuracy:.2f}")

print(f"Tuned XGBoost Accuracy: {accuracy_best:.2f}")

print("\nProgram Executed Successfully!")
