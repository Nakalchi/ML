(Below command downlod the pkg aslo run and install jupyter

python -m pip install seaborn
python -m pip install notebook
python -m pip install pandas numpy seaborn matplotlib
jupyter notebook)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'House_Size': [750, 800, 850, 900, 1000, 1100, 1200, 1300, 1400, 1500],
    'Price': [150000, 160000, 165000, 170000, 180000, 190000, 200000, 210000, 220000, 230000]
}

df = pd.DataFrame(data)

df.to_csv(
    r"C:\Users\Mandar\Downloads\house_prices.csv",
    index=False
)

print("Dataset:")
print(df)

dataset = pd.read_csv(
    r"C:\Users\Mandar\Downloads\house_prices.csv"
)

print("\nLoaded Dataset:")
print(dataset.head())

X = dataset[['House_Size']]
y = dataset['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining and Testing Data Sizes:")
print("Training Data Size:", X_train.shape[0])
print("Testing Data Size:", X_test.shape[0])

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Coefficients:")
print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

y_pred = model.predict(X_test)

print("\nPredictions on Test Data:")
print("Actual Prices:", y_test.values)
print("Predicted Prices:", y_pred)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.plot(X_train, model.predict(X_train), color='red', label='Regression Line')
plt.scatter(X_test, y_test, color='green', label='Test Data')

plt.title("Simple Linear Regression")
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price ($)")
plt.legend()

plt.show()
