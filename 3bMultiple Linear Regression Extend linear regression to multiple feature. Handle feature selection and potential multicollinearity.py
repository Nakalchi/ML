import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE

from statsmodels.stats.outliers_influence import variance_inflation_factor

data = pd.read_csv(r"C:\Users\Mandar\Downloads\cleaned_titanic.csv")

print(data.head())

print(data.info())
print(data.describe())

def calculate_vif(df):
    numeric_df = df.select_dtypes(include=np.number)
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).dropna()

    vif_data = pd.DataFrame()
    vif_data["feature"] = numeric_df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(numeric_df.values, i)
        for i in range(numeric_df.shape[1])
    ]

    return vif_data

X = data.drop("Survived", axis=1)
y = data["Survived"]

for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

imputer = SimpleImputer(strategy='mean')

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

print("VIF before handling multicollinearity:")
print(calculate_vif(X))

if 'X1' in X.columns:
    X = X.drop("X1", axis=1)
else:
    print("Column 'X1' not found in the DataFrame.")

print("VIF after handling multicollinearity:")
print(calculate_vif(X))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R^2: {r2}")

rfe = RFE(
    estimator=LinearRegression(),
    n_features_to_select=5
)

rfe.fit(X_train, y_train)

print("Selected Features:")
print(X.columns[rfe.support_])

plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()

residuals = y_test - y_pred

sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution")
plt.show()
