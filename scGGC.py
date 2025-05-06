from models import GCNAE, Generator, Discriminator, CellDataset
from preprocessing import preprocessing_csv, perform_pca
from training import train_gan, visualize_tsne, plot_clusters, enforce_label_consistency
from utils import build_adjacency_matrix
import torch
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import adjusted_rand_score
import time

# Function to get the top 30% nearest points to the centroid
def get_top_30_percent_nearest_centroids(Z_df, cluster_labels_df, n_clusters, top_percent):
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

def load_and_preprocess_data(expression_filename):
    df = pd.read_csv(expression_filename, index_col=0, header=0)
    processed_df = preprocessing_csv(df, expression_filename)
    processed_df.to_csv('./data/p1.csv', index=True, header=True)
    return pd.read_csv('./data/p1.csv', index_col=0, header=0)

def perform_pca_and_normalize(df):
    numeric_columns = df.select_dtypes(include=[float, int])
    normalized_data = numeric_columns.div(numeric_columns.sum())
    X2 = perform_pca(normalized_data, n_components=1300)
    X1 = normalized_data.T.dot(X2).astype(float)
    X3 = perform_pca(X1, n_components=800)
    return X1, X2, X3, normalized_data

def build_similarity_matrix(X3):
    B = pairwise_distances(X3, metric='euclidean')
    knn = NearestNeighbors(n_neighbors=10, metric='precomputed')
    knn.fit(B)
    knn_graph = knn.kneighbors_graph()
    return knn_graph.toarray()

def build_adjacency_and_normalize(normalized_data):
    W = build_adjacency_matrix(normalized_data)
    W = W.T
    row_sums = W.sum(axis=1)
    W_normalized = W.div(row_sums, axis=0)
    return W, W_normalized

def create_A_matrix(A1, W_normalized):
    alpha = 0.1
    beta = 1 - alpha
    weighted_A1 = alpha * A1
    weighted_W = beta * W_normalized
    sum_of_coefficients = alpha + beta
    weighted_A1 /= sum_of_coefficients
    weighted_W /= sum_of_coefficients

    zero_matrix1 = W_normalized.T
    identity_matrix1 = np.eye(2000)
    al = 0.1
    be = 1 - al
    zero_matrix = alpha * zero_matrix1
    identity_matrix = beta * identity_matrix1
    sum_of_coefficients = al + be
    zero_matrix /= sum_of_coefficients
    identity_matrix /= sum_of_coefficients

    A = np.block([[weighted_A1, weighted_W], [zero_matrix, identity_matrix]])
    return torch.tensor(A, dtype=torch.float32)

def initialize_model(X_combined_tensor, A_tensor):
    encoder_decoder_model = GCNAE(in_feats=X_combined_tensor.shape[1], n_hidden=64, n_layers=3, dropout=0.01)
    adj_rec, encoded = encoder_decoder_model(A_tensor, X_combined_tensor)
    return encoder_decoder_model

def train_and_evaluate_model(encoder_decoder_model, A_tensor, X_combined_tensor, labels, optimizer, loss_fn):
    for epoch in range(100):
        optimizer.zero_grad()
        adj_rec, _ = encoder_decoder_model(A_tensor, X_combined_tensor)
        loss = loss_fn(adj_rec, labels)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/100], Loss: {loss.item()}')

def extract_latent_variables(encoder_decoder_model, A_tensor, X_combined_tensor):
    _, embeddings = encoder_decoder_model(A_tensor, X_combined_tensor)
    latent_variables = embeddings.detach().numpy()
    latent_variables_df = pd.DataFrame(latent_variables)
    latent_variables_df.to_csv('./data/latent_variables.csv', index=False, header=False)
    return latent_variables

def clustering_analysis(latent_variables, df):
    cell_names = df.columns.tolist()
    Z_df = pd.DataFrame(latent_variables, index=cell_names)
    Z_df.to_csv('./data/Z_df.csv', header=False)
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(Z_df)
    cluster_labels = kmeans.labels_
    cluster_labels_df = pd.DataFrame(cluster_labels, index=cell_names, columns=['Cluster'])
    cluster_labels_df.index.name = "CellName"
    cluster_labels_df.to_csv('./data/scGGCn1.csv')
    return cluster_labels, Z_df

def compute_ari(true_labels, cluster_labels):
    ari = adjusted_rand_score(true_labels, cluster_labels)
    print(f'Adjusted Rand Index (ARI): {ari}')

def visualize_tsne_and_plot(Z_df, cluster_labels):
    tsne_plot_path = './data/scGGC66.png'
    tsne_csv_path = './data/GraphsctSNE_results.csv'
    plot_clusters(Z_df.values, cluster_labels, title='t-SNE Visualization of Cells', save_path=tsne_plot_path,
                  tsne_csv_path=tsne_csv_path)

def main():
    start_time_1 = time.time()

    expression_filename = './data/input.csv'
    df = load_and_preprocess_data(expression_filename)

    X1, X2, X3, normalized_data = perform_pca_and_normalize(df)

    A1 = build_similarity_matrix(X3)

    W, W_normalized = build_adjacency_and_normalize(normalized_data)

    A_tensor = create_A_matrix(A1, W_normalized)

    X_combined = np.vstack((X1, X2))
    X_combined_tensor = torch.tensor(X_combined, dtype=torch.float32)

    encoder_decoder_model = initialize_model(X_combined_tensor, A_tensor)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(encoder_decoder_model.parameters(), lr=0.01)
    labels = torch.tensor(A_tensor, dtype=torch.float32)

    train_and_evaluate_model(encoder_decoder_model, A_tensor, X_combined_tensor, labels, optimizer, loss_fn)

    latent_variables = extract_latent_variables(encoder_decoder_model, A_tensor, X_combined_tensor)

    cluster_labels, Z_df = clustering_analysis(latent_variables, df)

    true_labels_filename = './data/clusters.csv'
    true_labels_df = pd.read_csv(true_labels_filename, index_col=0)
    true_labels = true_labels_df['Cluster'].values

    compute_ari(true_labels, cluster_labels)

    visualize_tsne_and_plot(Z_df, cluster_labels)

    end_time_1 = time.time()
    elapsed_time_1 = end_time_1 - start_time_1
    print(f"Saving CSV file took {elapsed_time_1:.2f} seconds.")

if __name__ == '__main__':
    main()
