import pandas as pd
import numpy as np
import csv
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

training_data_raw = pd.read_csv("./Train.csv")

training_data_raw.drop('ID', axis=1, inplace=True)

X = pd.get_dummies(training_data_raw, columns=['Var_1', 'Spending_Score', 'Profession', 'Gender', 'Graduated', 'Ever_Married', 'Segmentation'])

imputer = SimpleImputer(strategy='most_frequent')
columns = X.columns
X[columns] = imputer.fit_transform(X[columns])

X = X.astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_3d = PCA(n_components=3)
X_3d = pca_3d.fit_transform(X_scaled)


dataset_pca = pd.DataFrame(abs(pca_3d.components_), columns=X.columns, index=['PC_1', 'PC_2', 'PC_3'])
print('\n\n', dataset_pca)

random_indices = np.random.choice(X_3d.shape[0], 200, replace=False)
X_3d_sample = X_3d[random_indices, :]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the 3D PCA results with randomly selected points
ax.scatter(X_3d_sample[:, 0], X_3d_sample[:, 1], X_3d_sample[:, 2], s=60)

ax.set_xlim(X_3d_sample[:, 0].min() - 1, X_3d_sample[:, 0].max() + 1)
ax.set_ylim(X_3d_sample[:, 1].min() - 1, X_3d_sample[:, 1].max() + 1)
ax.set_zlim(X_3d_sample[:, 2].min() - 1, X_3d_sample[:, 2].max() + 1)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# Setting title
ax.set_title('3D PCA Results (Sampled)')

# Show plot
plt.show()



inertias = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_3d)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()

# Update the optimal_num_clusters based on the elbow method result
optimal_num_clusters = 8
kmeans = KMeans(n_clusters=optimal_num_clusters, init='k-means++', random_state=42)
kmeans.fit(X_scaled)

# Predict cluster labels
# After fitting KMeans with the desired number of clusters
labels = kmeans.predict(X_scaled)

# Create a new figure for 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot points with their respective clusters
scatter = ax.scatter(X_3d_sample[:, 0], X_3d_sample[:, 1], X_3d_sample[:, 2], c=labels[random_indices], cmap='viridis', s=60, alpha=0.6)

# Set axis limits to accommodate the range of sampled data points
ax.set_xlim(X_3d_sample[:, 0].min() - 1, X_3d_sample[:, 0].max() + 1)
ax.set_ylim(X_3d_sample[:, 1].min() - 1, X_3d_sample[:, 1].max() + 1)
ax.set_zlim(X_3d_sample[:, 2].min() - 1, X_3d_sample[:, 2].max() + 1)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# Setting title
ax.set_title(f'KMeans Clustering with {optimal_num_clusters} clusters')

# Show color bar legend for clusters
plt.colorbar(scatter, ax=ax, label='Clusters')

# Show plot
plt.show()