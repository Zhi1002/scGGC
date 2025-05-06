import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

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
    elif normalize_weights == "per_cell":
        X1 = normalized_data / (np.sum(normalized_data, axis=1, keepdims=True) + 1e-6)

    non_zeros = X1[(row_idx, gene_idx)]
    D_sum = np.sum(X1, axis=1)

    adjacency_matrix = np.zeros((num_cells, num_genes))
    for r, c, w in zip(row_idx, gene_idx, non_zeros):
        adjacency_matrix[r, c] = w / D_sum[r]

    if edge_norm:
        for i in range(num_cells):
            row_sum = np.sum(adjacency_matrix[i])
            if row_sum > 0:
                adjacency_matrix[i] = adjacency_matrix[i] / row_sum

    cell_labels = ['cell_{}'.format(i) for i in range(num_cells)]
    gene_labels = ['gene_{}'.format(i) for i in range(num_genes)]
    adjacency_df = pd.DataFrame(adjacency_matrix, index=cell_labels, columns=gene_labels)

    return adjacency_df
