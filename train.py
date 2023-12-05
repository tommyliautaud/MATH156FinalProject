import pandas as pd
import numpy as np
import csv
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

training_data_raw = pd.read_csv("./Train.csv")

X = training_data_raw.copy()

map_var_1= {}

for i, item in enumerate(set(X['Var_1'])):
    map_var_1[item] = i

map_spending_score = {"Low" : 0, "Average" : 1, "High" : 2}
map_profession = {}

for i, item in enumerate(set(X['Profession'])):
        map_profession[item] = i

map_gender = {"Male" : 1, "Female" : 0}

map_graduated = {"Yes" : 1, "No" : 0}

map_ever_married = {"Yes" : 1, "No" : 0}

map_segmentation = {"A" : 1, "B" : 2, "C" : 3, "D" : 4}

X["Var_1"] = X["Var_1"].map(map_var_1)
X["Spending_Score"] = X["Spending_Score"].map(map_spending_score)
X["Profession"] = X["Profession"].map(map_profession)
X["Gender"] = X["Gender"].map(map_gender)
X["Graduated"] = X["Graduated"].map(map_graduated)
X["Ever_Married"] = X["Ever_Married"].map(map_ever_married)
X["Segmentation"] = X["Segmentation"].map(map_segmentation)

imputer = SimpleImputer(strategy='most_frequent')
columns = X.columns
X[columns] = imputer.fit_transform(X[columns])

X = X.astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

dataset_pca = pd.DataFrame(abs(pca_2d.components_), columns=X.columns, index=['PC_1', 'PC_2'])
print('\n\n', dataset_pca)


plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.8)
plt.title('Data in Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

inertias = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_2d)
    inertias.append(kmeans.inertia_)

optimal_num_clusters = 5
kmeans = KMeans(n_clusters=optimal_num_clusters)
kmeans.fit(X_scaled)
                                                                            

# After fitting KMeans with the desired number of clusters
labels = kmeans.labels_

# Scatter plot the clustered data
plt.figure(figsize=(8, 6))

for label in np.unique(labels):
    plt.scatter(X_2d[labels == label, 0], X_2d[labels == label, 1], label=f'Cluster {label}')

plt.title('KMeans Clustering Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
plt.scatter(X_2d[labels == label, 0], X_2d[labels == label, 1], label=f'Cluster {label}')
