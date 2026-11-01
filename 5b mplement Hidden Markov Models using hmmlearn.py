import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt

np.random.seed(42)

observations = np.random.choice(
    ['A', 'B', 'C'],
    size=100,
    p=[0.5, 0.3, 0.2]
)

hidden_states = np.random.choice(
    ['X', 'Y'],
    size=100,
    p=[0.6, 0.4]
)

data = pd.DataFrame({
    'Observations': observations,
    'Hidden States': hidden_states
})

print("Generated Data:")
print(data.head())

observation_mapping = {
    obs: idx
    for idx, obs in enumerate(np.unique(observations))
}

encoded_observations = np.array(
    [observation_mapping[obs] for obs in observations]
)

print("Observation Encoding:")
print(observation_mapping)

n_states = 2

n_observations = len(observation_mapping)

model = hmm.MultinomialHMM(
    n_components=n_states,
    random_state=42,
    n_iter=100,
    tol=0.01
)

start_probs = np.array([0.6, 0.4])

model.startprob_ = start_probs

trans_probs = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

model.transmat_ = trans_probs

emission_probs = np.array([
    [0.5, 0.4, 0.1],
    [0.2, 0.3, 0.5]
])

model.emissionprob_ = emission_probs

print("Start Probabilities:", model.startprob_)
print("Transition Matrix:", model.transmat_)
print("Emission Probabilities:", model.emissionprob_)

encoded_observations = encoded_observations.reshape(-1, 1)

model.fit(encoded_observations)

predicted_states = model.predict(encoded_observations)

print("Predicted States:")
print(predicted_states)

state_mapping = {
    0: 'X',
    1: 'Y'
}

predicted_state_labels = [
    state_mapping[state]
    for state in predicted_states
]

data['Predicted States'] = predicted_state_labels

print("Data with Predicted States:")
print(data.head())

plt.figure(figsize=(12, 6))

plt.plot(
    data['Observations'],
    label='Observations',
    marker='o',
    linestyle='-',
    alpha=0.7
)

plt.plot(
    data['Predicted States'],
    label='Predicted States',
    marker='x',
    linestyle='--',
    alpha=0.7
)

plt.legend()

plt.title("Observations and Predicted States")
plt.xlabel("Time")
plt.ylabel("Value")

plt.show()
