import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# UCI Wine dataset: first column is class label, remaining columns are features.
wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
wine = pd.read_csv(wine_url, header=None)
X = wine.iloc[:, 1:].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
labels = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2, random_state=42)
X_plot = pca.fit_transform(X_scaled)

plt.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, cmap="viridis")
plt.title("K-means Clustering on UCI Wine Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
