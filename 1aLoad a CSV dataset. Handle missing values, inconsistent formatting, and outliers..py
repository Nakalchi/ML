import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

print(data.head())

print("Missing values in each column:")
print(data.isnull().sum())

data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop(columns=['Cabin'], inplace=True)

print("\nAfter handling missing values:")
print(data.isnull().sum())

data['Sex'] = data['Sex'].str.lower().str.strip()

print("\nUnique values in 'Sex' column after formatting:")
print(data['Sex'].unique())

sns.boxplot(x=data['Fare'], color='skyblue')
plt.title('Boxplot of Fare')
plt.show()

Q1 = data['Fare'].quantile(0.25)
Q3 = data['Fare'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data['Fare'] = np.where(
    data['Fare'] > upper_bound,
    upper_bound,
    np.where(data['Fare'] < lower_bound, lower_bound, data['Fare'])
)

sns.boxplot(x=data['Fare'], color='lightgreen')
plt.title('Boxplot of Fare (After Handling Outliers)')
plt.show()

data.to_csv(
    r"C:\Users\Mandar\Downloads\cleaned_titanic.csv",
    index=False
)

print("\nCleaned dataset saved as 'C:\\Users\\Mandar\\Downloads\\cleaned_titanic.csv'")
