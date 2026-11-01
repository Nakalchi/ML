import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
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

print("Loaded Dataset:")
print(data.head())

# Preprocessing if using custom CSV (like Titanic)
if response == "yes":
    # Automatically determine target: typically the 'Survived' column if present, otherwise the last column
    if 'Survived' in data.columns:
        target_col = 'Survived'
    else:
        target_col = data.columns[-1]
    
    print(f"\nUsing '{target_col}' as the target column.")
    
    # Drop columns that are unique identifiers or have too many text values (e.g., Name, Ticket, PassengerId)
    cols_to_drop = [col for col in ['PassengerId', 'Name', 'Ticket'] if col in data.columns and col != target_col]
    data = data.drop(columns=cols_to_drop)
    
    # Handle missing values
    for col in data.columns:
        if data[col].isnull().any():
            if data[col].dtype == 'object':
                data[col] = data[col].fillna(data[col].mode()[0])
            else:
                data[col] = data[col].fillna(data[col].median())
                
    # Encode categorical features and target
    le = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = le.fit_transform(data[col].astype(str))
            
    X = data.drop(columns=[target_col]).values
    y = data[target_col].values
else:
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}\n")

print("Correct Predictions:")
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        print(f"Sample {i}: Predicted={y_pred[i]}, Actual={y_test[i]}")

print("\nIncorrect Predictions:")
for i in range(len(y_test)):
    if y_pred[i] != y_test[i]:
        print(f"Sample {i}: Predicted={y_pred[i]}, Actual={y_test[i]}")
