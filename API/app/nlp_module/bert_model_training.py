import pandas as pd
from sklearn.cluster import KMeans

# Determine optimal number of clusters using the Elbow Method
# import matplotlib.pyplot as plt
# inertia = []
# for k in range(1, 10):
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
#     kmeans.fit(features)
#     inertia.append(kmeans.inertia_)
#
# # Plot the Elbow Method graph
# plt.plot(range(1, 10), inertia, marker='o')
# plt.xlabel("Number of clusters")
# plt.ylabel("Inertia")
# plt.title("Elbow Method for Optimal Clusters")
# plt.show()

df = pd.read_csv("data/merged_dataset.csv")

# Select numerical features (excluding meal_description)
features = df[['carb', 'protein', 'fat', 'energy']]

# Train K-Means with k=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(features)

# Display the first few rows to see assigned clusters
print(df.head())

# Count the number of meals in each cluster
print(df['cluster'].value_counts())

# Display cluster centroids
print(pd.DataFrame(kmeans.cluster_centers_, columns=['carb', 'protein', 'fat', 'energy']))


