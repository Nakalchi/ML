(Below command downlod the pkg aslo run and install jupyter

python -m pip install seaborn
python -m pip install notebook
python -m pip install pandas numpy seaborn matplotlib
jupyter notebook)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
data = pd.read_csv(url)

print("First 5 rows of the dataset:")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nDescriptive Statistics for Numerical Columns:")
print(data.describe())

print("\nUnique values in 'species' column:")
print(data['species'].value_counts())

data.hist(figsize=(10, 8), color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Numerical Features")
plt.show()

sns.countplot(x='species', data=data, palette='pastel')
plt.title("Count of Each Species")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(data['sepal_length'], data['sepal_width'], alpha=0.7, c='blue')
plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()

sns.pairplot(data, hue='species', palette='husl', diag_kind='kde')
plt.suptitle("Pairplot of Features by Species", y=1.02)
plt.show()

sns.boxplot(x='species', y='petal_length', data=data, palette='Set3')
plt.title("Boxplot of Petal Length by Species")
plt.show()

features = data.drop(columns=['species'])

target = data['species']

print("\nFeatures:")
print(features.head())

print("\nTarget:")
print(target.head())

sns.countplot(x=target, palette='viridis')
plt.title("Target Variable Distribution")
plt.show()

data.to_csv(
    r"C:\Users\Mandar\Downloads\processed_iris.csv",
    index=False
)

print("\nProcessed dataset saved as 'C:\\Users\\Mandar\\Downloads\\processed_iris.csv'")
