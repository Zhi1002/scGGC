import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import coo_matrix

def build_adjacency_matrix(normalized_data, threshold=0, normalize_weights="log_per_cell", edge_norm=True):
    if isinstance(normalized_data, pd.DataFrame):
        normalized_data = normalized_data.to_numpy()
    num_cells, num_genes = normalized_data.shape
    row_idx, gene_idx = np.nonzero(normalized_data > threshold)
    if normalize_weights == "none":
        X1 = normalized_data
    elif normalize_weights == "log_per_cell":
        X1 = np.log1p(normalized_data)
        X1 = X1 / (np.sum(X1, axis=1, keepdims=True) + 1e-6)
    else:
        X1 = normalized_data / (np.sum(normalized_data, axis=1, keepdims=True) + 1e-6)
    non_zeros = X1[(row_idx, gene_idx)]
    D_sum = np.sum(X1, axis=1)
    adjacency = np.zeros((num_cells, num_genes))
    for r, c, w in zip(row_idx, gene_idx, non_zeros):
        adjacency[r, c] = w / D_sum[r]
    if edge_norm:
        adjacency = adjacency / (adjacency.sum(axis=1, keepdims=True) + 1e-6)
    cells = [f'cell_{i}' for i in range(num_cells)]
    genes = [f'gene_{i}' for i in range(num_genes)]
    return pd.DataFrame(adjacency, index=cells, columns=genes)