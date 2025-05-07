import numpy as np
from sklearn.metrics import pairwise_distances


def get_top_30_percent_nearest_centroids(Z_df, cluster_labels_df, n_clusters, top_percent=0.3):
    top_30_percent_indices = []
    for cluster in range(n_clusters):
        cluster_indices = np.where(cluster_labels_df['Cluster'] == cluster)[0]
        cluster_points = Z_df.iloc[cluster_indices]
        centroid = cluster_points.mean(axis=0)
        distances = pairwise_distances(cluster_points, centroid.values.reshape(1, -1), metric='euclidean').flatten()
        top_30_percent_count = int(len(distances) * top_percent)
        top_30_percent_idx = np.argsort(distances)[:top_30_percent_count]
        top_30_percent_indices.extend(cluster_indices[top_30_percent_idx])
    return top_30_percent_indices