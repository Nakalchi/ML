(Below command downlod the pkg aslo run and install jupyter

python -m pip install seaborn
python -m pip install notebook
python -m pip install pandas numpy seaborn matplotlib
jupyter notebook)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from google.colab import files
print("Do you have a CSV file to upload? (yes/no)")
response = input().lower()
if response == "yes":
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
else:
    np.random.seed(42)

    X1 = np.random.normal(
        loc=0,
        scale=1,
        size=(300, 2)
    )
    X2 = np.random.normal(
        loc=5,
        scale=1,
        size=(300, 2)
    )
    X = np.vstack([X1, X2])
    data = pd.DataFrame(
        X,
        columns=["Feature_1", "Feature_2"]
    )
    filename = "synthetic_data.csv"
    data.to_csv(filename, index=False)
    print(f"Synthetic dataset saved as {filename}.")
data = pd.read_csv(filename)
print("Dataset Preview:")
print(data.head())
if response == "yes":
    # Keep only numeric columns for GMM fitting and plotting
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        raise ValueError("Uploaded dataset must have at least 2 numerical columns for visualization.")
    # Use all numeric columns for clustering, but the first two for 2D plots
    plot_x = numeric_cols[0]
    plot_y = numeric_cols[1]
    fit_data = data[numeric_cols]
else:




    plot_x = "Feature_1"
    plot_y = "Feature_2"
    fit_data = data
sns.scatterplot(
    data=data,
    x=plot_x,
    y=plot_y
)
plt.title("Input Data for Clustering")
plt.xlabel(plot_x)
plt.ylabel(plot_y)
plt.show()
n_components = 2
gmm = GaussianMixture(
    n_components=n_components,
    covariance_type='full',
    random_state=42
)
gmm.fit(fit_data)
labels = gmm.predict(fit_data)
data['Cluster'] = labels
sns.scatterplot(
    data=data,
    x=plot_x,
    y=plot_y,
    hue="Cluster",
    palette="viridis",
    marker="o"
)
plt.title("Gaussian Mixture Model Clustering")
plt.xlabel(plot_x)
plt.ylabel(plot_y)
plt.legend()
plt.show()
if response == "yes":
    idx_x = numeric_cols.index(plot_x)
    idx_y = numeric_cols.index(plot_y)
    means = gmm.means_[:, [idx_x, idx_y]]
    covariances = gmm.covariances_[:, [idx_x, idx_y]][:, :, [idx_x, idx_y]]
else:
    means = gmm.means_
    covariances = gmm.covariances_

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=data,
    x=plot_x,
    y=plot_y,
    hue="Cluster",
    palette="viridis",





    marker="o",
    s=60,
    alpha=0.7
)

for mean, covar in zip(means, covariances):
    v, w = np.linalg.eigh(covar)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = angle * 180.0 / np.pi

    ellipse = plt.matplotlib.patches.Ellipse(
        mean,
        v[0],
        v[1],
        angle=angle,
        color='red',
        alpha=0.3
    )
    plt.gca().add_patch(ellipse)
plt.title("GMM Clustering with Gaussian Components")
plt.xlabel(plot_x)
plt.ylabel(plot_y)
plt.legend()
plt.show()

log_likelihood = gmm.score(fit_data)
print(f"Log-Likelihood of the data: {log_likelihood:.2f}")

# Predict clusters for sample test points
if response == "yes":
    # Generate a test point matching the number of numeric dimensions
    num_features = fit_data.shape[1]
    new_data = np.tile(fit_data.median().values, (3, 1))
    # Perturb the sample points slightly to simulate different entities
    new_data[0] *= 0.9
    new_data[2] *= 1.1
else:
    new_data = np.array([
        [1.5, 2.5],
        [4.5, 5.5],
        [7.0, 8.0]
    ])

new_labels = gmm.predict(new_data)
print("Predicted Clusters for New Data Points:")
for i, label in enumerate(new_labels):
    print(f"Data point {new_data[i]} is in Cluster {label}")
