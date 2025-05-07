import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from processing import preprocessingCSV, perform_pca
from graph import build_adjacency_matrix
from select import get_top_30_percent_nearest_centroids
from untils import check_and_create_paths
from model import GCNAE,train_ac_gan, extract_embeddings_and_predict_labels


def perform_pca1(X1, n_components):
    numeric_columns = X1.iloc[:, 1:].select_dtypes(include=[float, int])
    mean = np.mean(numeric_columns, axis=1)
    standardized_data = (X1.iloc[:, 1:].T - mean) / np.std(numeric_columns, axis=1, ddof=1)
    standardized_data = np.nan_to_num(standardized_data, nan=0.0, posinf=0.0, neginf=0.0)
    cov_matrix = np.cov(standardized_data, rowvar=True)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    selected = eigenvectors[:, idx]
    return np.dot(standardized_data.T, selected)


if __name__ == "__main__":
    # check_and_create_paths()
    start_time_1 = time.time()

    # # 数据预处理
    # df = pd.read_csv('E:/jsj/user06/zz/EMATB7678/EMATB7678.csv', index_col=0, header=0)
    # processed_df = preprocessingCSV(df)
    # processed_df.to_csv('E:/jsj/user06/zz/EMATB7678/ceshi/p1.csv', index=True, header=True)

    # 读取预处理结果
    df = pd.read_csv('E:/jsj/user06/zz/EMATB7678/ceshi/p1.csv', index_col=0, header=0)

    numeric_columns = df.select_dtypes(include=[float, int])
    normalized_data = numeric_columns.div(numeric_columns.sum())

    # PCA 和邻接矩阵构建
    X2 = perform_pca(normalized_data, n_components=800)
    pca_df = pd.DataFrame(X2)
    print("X2的形状:", X2.shape)

    X1 = normalized_data.T.dot(X2).astype(float)
    print(X1)
    print("X1的形状:", X1.shape)

    X3 = perform_pca1(X1, n_components=100)
    B = pairwise_distances(X3, metric='euclidean')
    print(B)
    print("B的形状:", B.shape)

    start_time_2 = time.time()

    # # 使用KNN算法获取细胞间的邻接图
    # knn = NearestNeighbors(n_neighbors=10, metric='precomputed')
    # knn.fit(B)# 使用距离矩阵B进行KNN训练
    # # 获取K个最近邻居的图表示
    # knn_graph = knn.kneighbors_graph()
    # # 注意：这将创建一个较大的矩阵，可能占用大量内存
    # A1 = knn_graph.toarray()
    k = 10
    knn = NearestNeighbors(n_neighbors=k, metric='precomputed')
    knn.fit(B)

    # Step 2. 获取每个点的k近邻（返回的是索引和距离）
    distances, indices = knn.kneighbors(B)

    # Step 3. 使用高斯核函数生成权重
    sigma = np.mean(distances)  # 也可以手动指定，比如 sigma=0.5
    weights = np.exp(- (distances ** 2) / (2 * sigma ** 2))

    # Step 4. 生成稀疏邻接矩阵（只保留k近邻）
    rows = np.repeat(np.arange(B.shape[0]), k)  # 每个点重复k次
    cols = indices.flatten()
    data = weights.flatten()

    A_weighted_sparse = coo_matrix((data, (rows, cols)), shape=B.shape)

    # 如果需要dense格式
    A_weighted_dense = A_weighted_sparse.toarray()

    # Step 5. （可选）行归一化

    A1_normalized = normalize(A_weighted_dense, norm='l1', axis=1)
    end_time_2 = time.time()
    elapsed_time_2 = end_time_2 - start_time_2
    print(f"Saving CSV file took {elapsed_time_2:.2f} seconds.")

    # # 如果需要稀疏格式的邻接矩阵以节省内存
    # (i, j) = knn_graph.nonzero()
    # A1_sparse = coo_matrix((np.ones_like(i), (i, j)), shape=knn_graph.shape)
    # # 如果需要将稀疏矩阵转换为普通矩阵格式，可以使用toarray()方法
    # A1_full = A1_sparse.toarray()
    # print(A1_full)
    #
    # # 对邻接矩阵进行行归一化处理
    # row_sums = A1_full.sum(axis=1)
    # A1_normalized = A1_full / row_sums[:, np.newaxis]
    print(A1_normalized)

    # # 将邻接矩阵 A1_normalized 存储为 CSV 文件
    # pd.DataFrame(A1_normalized).to_csv('E:/jsj/user06/zz/GSE138852/GSE138852/GSE138852_counts/A1_normalized.csv'
    #                                    , index=False, header=False)

    start_time_3 = time.time()
    W = build_adjacency_matrix(normalized_data)
    W = W.T
    # # # 打印邻接矩阵的形状
    W.to_csv('E:/jsj/user06/zz/EMATB7678/ceshi/W_original.csv', index=False, header=False)
    end_time_3 = time.time()
    elapsed_time_3 = end_time_3 - start_time_3
    print(f"3 file took {elapsed_time_3:.2f} seconds.")

    # # # 读取原始的邻接矩阵文件
    W_original = pd.read_csv('E:/jsj/user06/zz/EMATB7678/ceshi/W_original.csv', header=None)

    # 对邻接矩阵进行行归一化处理
    # row_sums = W_original.sum(axis=1)
    # W_normalized = W_original.div(row_sums, axis=0)
    W_normalized = normalize(W_original, norm='l1', axis=1)

    ## 添加权重系数
    alpha = 0.2  # 调整 alpha 的值，使得 A1 和 W 的系数之和为 1
    beta = 1 - alpha  # 调整 beta 的值
    weighted_A1 = alpha * A1_normalized
    weighted_W = beta * W_normalized

    # 创建单位矩阵
    identity_matrix1 = np.eye(2000)

    # # 创建零矩阵
    # zero_matrix = np.zeros((2000, 4449))
    # # 创建单位矩阵
    # identity_matrix = np.eye(2000)

    # Print the shapes of the matrices
    print("Shape of weighted_A1:", weighted_A1.shape)
    print("Shape of weighted_W:", weighted_W.shape)
    print("Shape of identity_matrix:", identity_matrix1.shape)

    # 合并矩阵
    A = np.block([[weighted_A1, weighted_W], [beta * W_normalized.T, 0 * identity_matrix1]])
    A_tensor = torch.tensor(A, dtype=torch.float32)
    print(A)
    print("A的形状:", A.shape)

    # 将 NumPy 数组转换为 DataFrame
    A_df = pd.DataFrame(A)
    A_df.to_csv('E:/jsj/user06/zz/EMATB7678/ceshi/A_matrix.csv', index=False)

    # 合并 X1 和 X2
    X_combined = np.vstack((X1, X2))
    print("X_combined的形状：", X_combined.shape)
    # 将 X_combined 转换为 PyTorch 张量
    X_combined_tensor = torch.tensor(X_combined, dtype=torch.float32)

    encoder_decoder_model = GCNAE(in_feats=X_combined_tensor.shape[1],
                                  n_hidden=64,
                                  n_layers=3,
                                  dropout=0.01)

    adj_rec, encoded = encoder_decoder_model(A_tensor, X_combined_tensor)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(encoder_decoder_model.parameters(), lr=0.001)
    labels = torch.tensor(A, dtype=torch.float32)


    def train_model(model, A, features, labels, optimizer, loss_fn, num_epochs=100):
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            adj_rec, _ = model(A, features)
            loss = loss_fn(adj_rec, labels)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')


    start_time_4 = time.time()
    train_model(encoder_decoder_model, A_tensor, X_combined_tensor, labels, optimizer, loss_fn, num_epochs=70)
    end_time_4 = time.time()
    elapsed_time_4 = end_time_4 - start_time_4
    print(f"3 file took {elapsed_time_4:.2f} seconds.")

    _, embeddings = encoder_decoder_model(A_tensor, X_combined_tensor)
    latent_variables = embeddings.detach().numpy()
    latent_variables_df = pd.DataFrame(latent_variables)
    latent_variables_df.to_csv('E:/jsj/user06/zz/EMATB7678/ceshi/latent_variables.csv', index=False, header=False)
    latent_variables = latent_variables[:6666]

    cell_names = df.columns.tolist()
    Z_df = pd.DataFrame(latent_variables, index=cell_names)
    Z_df.to_csv('E:/jsj/user06/zz/EMATB7678/ceshi/Z_df.csv', header=False)

    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(Z_df)
    cluster_labels = kmeans.labels_
    cluster_labels_df = pd.DataFrame(cluster_labels, index=cell_names, columns=['Cluster'])
    cluster_labels_df.index.name = "CellName"
    cluster_labels_df.to_csv('E:/jsj/user06/zz/EMATB7678/ceshi/scGGC.csv')

    # 读取真实标签
    true_labels_filename = 'E:/jsj/user06/zz/EMATB7678/clusters.csv'
    true_labels_df = pd.read_csv(true_labels_filename, index_col=0)
    true_labels = true_labels_df['Cluster8'].values

    # 计算并打印ARI
    ari = adjusted_rand_score(true_labels, cluster_labels)
    print(f'Adjusted Rand Index (ARI): {ari}')
    # 获取前30%距离质心最近的点的索引
    top_30_percent_indices = get_top_30_percent_nearest_centroids(Z_df, cluster_labels_df, n_clusters=8,
                                                                  top_percent=0.4)
    # 提取这些点的细胞名和标签
    top_n_percent_df = cluster_labels_df.iloc[top_30_percent_indices]
    top_n_percent_df.index.name = "CellName"
    top_n_percent_df.to_csv('E:/jsj/user06/zz/EMATB7678/ceshi/selected_cells.csv', index=True, header=True)

    # GAN 部分
    # 读取预处理后的数据和标签
    selected_cells_df = pd.read_csv('E:/jsj/user06/zz/EMATB7678/ceshi/selected_cells.csv')
    selected_cell_names = selected_cells_df['CellName'].tolist()
    gene_expression_df = df.T

    initial_labels_df = pd.read_csv('E:/jsj/user06/zz/EMATB7678/ceshi/scGGC.csv')
    initial_label_map = initial_labels_df.set_index('CellName')['Cluster'].to_dict()

    real_labels_df = pd.read_csv('E:/jsj/user06/zz/EMATB7678/clusters.csv')
    real_label_map = real_labels_df.set_index('CellName')['Cluster8'].to_dict()
    all_true_labels = [real_label_map[cell] for cell in gene_expression_df.index]

    # 构建训练集 & 全部数据
    train_df = gene_expression_df.loc[selected_cell_names]

    # 标准化
    scaler = MinMaxScaler()
    all_scaled = scaler.fit_transform(gene_expression_df.values)
    train_scaled = scaler.transform(train_df.values)

    # 转为 PyTorch Tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.tensor(train_scaled, dtype=torch.float32).to(device)
    X_all = torch.tensor(all_scaled, dtype=torch.float32).to(device)

    # 训练标签
    y_train_raw = [initial_label_map[cell] for cell in selected_cell_names]
    unique_labels = np.unique(y_train_raw)
    label2idx = {lab: i for i, lab in enumerate(unique_labels)}
    y_train = torch.tensor([label2idx[lab] for lab in y_train_raw], dtype=torch.long).to(device)
    num_classes = len(unique_labels)

    # 训练AC-GAN
    G, D = train_ac_gan(X_train, y_train, device, num_classes, X_train.shape[1], n_epochs=30)

    # 提取嵌入向量 & 预测标签
    embeddings, predicted_idx, probabilities = extract_embeddings_and_predict_labels(D, X_all, device, num_classes)

    # 检查并补全“缺失的簇”
    counts = np.bincount(predicted_idx, minlength=num_classes)
    missing = np.where(counts == 0)[0]

    while missing.size > 0:
        counts = np.bincount(predicted_idx, minlength=num_classes)
        for c in missing:
            candidates = np.where(counts[predicted_idx] > 1)[0]
            if candidates.size == 0:
                break
            # 在这些候选里，选出对类 c 预测概率最高的样本
            j = candidates[np.argmax(probabilities[candidates, c])]  # 使用 probabilities
            # 重新分配它到缺失类 c
            old_c = predicted_idx[j]
            predicted_idx[j] = c
            counts[old_c] -= 1
            counts[c] += 1
        # 更新缺失列表
        missing = np.where(counts == 0)[0]

    # 将索引映射回初始聚类的真实标签值
    idx2label = {v: k for k, v in label2idx.items()}
    predicted_labels = [idx2label[i] for i in predicted_idx]

    # 保存嵌入向量
    Z_df = pd.DataFrame(embeddings, index=gene_expression_df.index,
                        columns=[f"dim{i}" for i in range(embeddings.shape[1])])


    # 可视化 & 计算 ARI
    def plot_clusters(features, labels, title, save_path):
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(features)
        tsne_df = pd.DataFrame({
            'tsne_dim1': tsne_results[:, 0],
            'tsne_dim2': tsne_results[:, 1],
            'label': labels
        }, index=gene_expression_df.index)
        tsne_df.to_csv(tsne_csv_path)
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='tsne_dim1', y='tsne_dim2', hue='label', palette='nipy_spectral', data=tsne_df, legend='full',
                        alpha=0.8, s=35)
        plt.title(title)
        plt.savefig(save_path)
        plt.show()


    tsne_csv_path = 'E:/jsj/user06/zz/EMATB7678/ceshi/gantsne1.csv'
    tsne_plot_path = 'E:/jsj/user06/zz/EMATB7678/ceshi/gan1.png'
    ari = adjusted_rand_score(all_true_labels, predicted_labels)
    import matplotlib

    matplotlib.use('TkAgg')  # 强制使用 TkAgg 后端
    import matplotlib.pyplot as plt

    plot_clusters(Z_df.values, predicted_labels, title=f"t-SNE Visualization of Cells (ARI={ari:.4f})",
                  save_path=tsne_plot_path)

    print(f"Adjusted Rand Index (vs. real clusters): {ari:.4f}")
    cell_names = gene_expression_df.T.columns.tolist()
    cluster_labels_df = pd.DataFrame(predicted_labels, index=cell_names, columns=['Cluster'])
    cluster_labels_df.index.name = "CellName"
    cluster_labels_df.to_csv('E:/jsj/user06/zz/EMATB7678/scGGCr/scGGCGan1.csv')