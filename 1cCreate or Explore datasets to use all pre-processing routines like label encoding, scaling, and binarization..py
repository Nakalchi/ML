(Below command downlod the pkg aslo run and install jupyter

python -m pip install seaborn
python -m pip install notebook
python -m pip install pandas numpy seaborn matplotlib
jupyter notebook)
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, Binarizer

data = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'A', 'B', 'C'],
    'Age': [23, 45, 31, 22, 35, 30],
    'Income': [50000, 60000, 70000, 80000, 90000, 100000],
    'Has_Car': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No']
})

print("Sample Dataset:")
print(data)

label_encoder = LabelEncoder()

data['Category_Encoded'] = label_encoder.fit_transform(data['Category'])
data['Has_Car_Encoded'] = label_encoder.fit_transform(data['Has_Car'])

print("\nAfter Label Encoding:")
print(data)

min_max_scaler = MinMaxScaler()
data['Income_MinMax'] = min_max_scaler.fit_transform(data[['Income']])

standard_scaler = StandardScaler()
data['Age_Standardized'] = standard_scaler.fit_transform(data[['Age']])

print("\nAfter Scaling:")
print(data)

binarizer = Binarizer(threshold=75000)
data['Income_Binary'] = binarizer.fit_transform(data[['Income']])

print("\nAfter Binarization:")
print(data)

data.to_csv(
    r"C:\Users\Mandar\Downloads\processed_data.csv",
    index=False
)

print("\nProcessed dataset saved as 'C:\\Users\\Mandar\\Downloads\\processed_data.csv'")
