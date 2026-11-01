import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Open file dialog
Tk().withdraw()

filename = askopenfilename(
    title="Select CSV File",
    filetypes=[("CSV Files", "*.csv")]
)

if not filename:
    print("No file selected!")
    exit()

# Load dataset
data = pd.read_csv(filename)

print("\nLoaded Dataset:")
print(data.head())

# Features (all columns except last)
X = data.iloc[:, :-1].values

# Target (last column)
y = data.iloc[:, -1]

# Convert text labels to numeric labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create KNN model
knn = KNeighborsClassifier(n_neighbors=3)

# Train model
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2f}")

# Show Results
print("\nPrediction Results:")
print("-" * 50)

for i in range(len(y_test)):
    predicted_label = encoder.inverse_transform([y_pred[i]])[0]
    actual_label = encoder.inverse_transform([y_test[i]])[0]

    result = "✓ Correct" if y_pred[i] == y_test[i] else "✗ Incorrect"

    print(
        f"Sample {i+1}: "
        f"Predicted = {predicted_label}, "
        f"Actual = {actual_label} "
        f"({result})"
    )

# Count correct and incorrect predictions
correct = sum(y_pred == y_test)
incorrect = len(y_test) - correct

print("\nSummary")
print("-" * 50)
print("Total Test Samples :", len(y_test))
print("Correct Predictions:", correct)
print("Wrong Predictions  :", incorrect)
print(f"Accuracy           : {accuracy:.2%}")
