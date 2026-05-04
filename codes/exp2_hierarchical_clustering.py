import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


# UCI Abalone dataset with one categorical column (sex) and numeric measurements.
abalone_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
columns = [
	"sex",
	"length",
	"diameter",
	"height",
	"whole_weight",
	"shucked_weight",
	"viscera_weight",
	"shell_weight",
	"rings",
]
abalone = pd.read_csv(abalone_url, header=None, names=columns)
abalone["sex"] = abalone["sex"].map({"M": 0, "F": 1, "I": 2})
X = abalone.drop(columns=["rings"]).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

agg_cluster = AgglomerativeClustering(n_clusters=3)
labels = agg_cluster.fit_predict(X_scaled)

# Use a subset for a readable dendrogram.
X_sample = X_scaled[:300]
linkage_matrix = linkage(X_sample, method="ward")
dendrogram(linkage_matrix)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="viridis", s=10)
plt.title("Hierarchical Clustering on UCI Abalone Dataset")
plt.xlabel("Scaled Feature 1")
plt.ylabel("Scaled Feature 2")
plt.show()
