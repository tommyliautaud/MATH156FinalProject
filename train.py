import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

training_data_raw = pd.read_csv("./Train.csv")

training_data_raw.drop(['ID', 'Profession', 'Var_1', 'Segmentation'], axis=1, inplace=True)

X = pd.DataFrame(training_data_raw)


spending_score_map = {'Low': 0, 'Medium': 1, 'High': 2}
X['Spending_Score'] = X['Spending_Score'].fillna('Unknown').map(spending_score_map)

# Handle missing values and map Gender
gender_map = {'Male': 0, 'Female': 1}  # Assuming 'Other' as an additional category
X['Gender'] = X['Gender'].fillna('Unknown').map(gender_map)

# Handle missing values and map Ever_Married and Graduated
married_map = {'Yes': 1, 'No': 0}
graduated_map = {'Yes': 1, 'No': 0}
X['Ever_Married'] = X['Ever_Married'].fillna('Unknown').map(married_map)
X['Graduated'] = X['Graduated'].fillna('Unknown').map(graduated_map)

X.dropna(inplace=True)

print(X)

X[X.columns] = StandardScaler().fit_transform(X)

pca_3d = PCA(n_components=3)

pca_3d_result = pca_3d.fit_transform(X)

print('Explained variation per principal component: {}'.format(pca_3d.explained_variance_ratio_))

# >> Explained variation per principal component: [0.36198848 0.1920749 ]

print('Cumulative variance explained by 2 principal components: {:.2%}'.format(np.sum(pca_3d.explained_variance_ratio_)))

dataset_pca = pd.DataFrame(abs(pca_3d.components_), columns=X.columns, index=['PC_1', 'PC_2', 'PC_3'])
print('\n\n', dataset_pca)

print("\n*************** Most important features *************************")
print('As per PC 1:\n', (dataset_pca[dataset_pca > 0.4].iloc[0]).dropna())   
print('\n\nAs per PC 2:\n', (dataset_pca[dataset_pca > 0.4].iloc[1]).dropna())
print('\n\nAs per PC 2:\n', (dataset_pca[dataset_pca > 0.4].iloc[2]).dropna())
print("\n******************************************************************")

# Your previous code for PCA and data preprocessing

# Plotting a random sample after PCA transformation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Randomly select a subset of indices for visualization
random_indices = np.random.choice(X.shape[0], 500, replace=False)

# Sample the PCA transformed data
pca_sample = pca_3d_result[random_indices, :]

# Plotting the 3D scatter plot
ax.scatter(pca_sample[:, 0], pca_sample[:, 1], pca_sample[:, 2], s=60)

# Set labels for each axis
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# Setting title
ax.set_title('3D Scatter Plot after PCA')

# Show plot
plt.show()

silhouette_scores = []
for k in range(2, 11):  # K varies from 2 to 10
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(pca_3d_result)  # Using PCA transformed data
    silhouette_avg = silhouette_score(pca_3d_result, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plotting silhouette scores
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.show()

# Assuming optimal_num_clusters is determined using silhouette analysis or any other method
optimal_num_clusters = 4  # Replace this with the actual optimal number of clusters

# Initialize KMeans with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(pca_3d_result)  # Fit KMeans on the PCA transformed data

# Plotting the clusters in 3D space
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting the clusters using the PCA transformed data
ax.scatter(pca_3d_result[:, 0], pca_3d_result[:, 1], pca_3d_result[:, 2],
           c=cluster_labels, cmap='viridis', s=60, alpha=0.6)

# Set axis labels
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# Set title for the plot
ax.set_title(f'KMeans Clustering with {optimal_num_clusters} clusters')

# Show the plot
plt.show()


