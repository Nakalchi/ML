import pandas as pd
import numpy as np

data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny', 'Rainy'],
    'Temperature': ['Warm', 'Cold', 'Warm', 'Warm', 'Cold'],
    'Humidity': ['Normal', 'High', 'High', 'Normal', 'Normal'],
    'Wind': ['Strong', 'Strong', 'Weak', 'Strong', 'Weak'],
    'Water': ['Warm', 'Warm', 'Cool', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Same', 'Change'],
    'Condition': ['Yes', 'No', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)

df.to_csv(
    r"C:\Users\Mandar\Downloads\training_data.csv",
    index=False
)

print("Dataset:")
print(df)

dataset = pd.read_csv(
    r"C:\Users\Mandar\Downloads\training_data.csv"
)

print("\nLoaded Dataset:")
print(dataset)

def find_s(training_data):
    features = training_data.iloc[:, :-1].values
    target = training_data.iloc[:, -1].values

    hypothesis = ['Ø'] * features.shape[1]

    for i, example in enumerate(features):
        if target[i] == 'Yes':
            for j in range(len(hypothesis)):
                if hypothesis[j] == 'Ø':
                    hypothesis[j] = example[j]
                elif hypothesis[j] != example[j]:
                    hypothesis[j] = '?'

    return hypothesis

final_hypothesis = find_s(dataset)

print("\nFinal Specific Hypothesis:")
print(final_hypothesis)

