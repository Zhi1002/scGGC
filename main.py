import torch
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.sparse import coo_matrix
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score
import time



# 数据预处理函数
def preprocessingCSV(df, expressionFilename=None):
    # 如果传入了DataFrame，则直接使用该DataFrame进行处理
    if isinstance(df, pd.DataFrame):
        data = df
    else:
        # 否则，从CSV文件读取数据
        data = pd.read_csv(expressionFilename, index_col=0, header=0)

    # 把所有细胞中，表达比例还不到1%的基因去除掉。
    data = data[data[data.columns[1:]].astype('bool').mean(axis=1) >= 0.01]
    print('After preprocessing, {} genes remaining'.format(data.shape[0] - 1))

    # # 把基因表达不到1%的细胞去除掉。
    # data = data[data.columns[data.iloc[1:, :].astype('bool').mean(axis=0) >= 0.01]]
    # print('After preprocessing, {} cells remaining'.format(data.shape[1] - 1))

    # 取出离基因均值最远的2000个基因。
    data = data.loc[(data.iloc[1:, 1:].var(axis=1, numeric_only=True).sort_values()[-2000:]).index]
    data.fillna(0, inplace=True)

    return data

def perform_pca(normalized_data, n_components):
    # 仅选择数值列进行均值计算
    numeric_columns = normalized_data.iloc[:, 1:].select_dtypes(include=[float, int])

    # 计算数值列的均值
    mean = np.mean(numeric_columns, axis=1)

    # 使用计算得到的均值对数据进行标准化
    standardized_data = (normalized_data.iloc[:, 1:].T - mean) / np.std(numeric_columns, axis=1, ddof=1)

    if np.any(np.isinf(standardized_data)) or np.any(np.isnan(standardized_data)):
        standardized_data = np.nan_to_num(standardized_data, nan=0.0, posinf=0.0, neginf=0.0)

    # 计算协方差矩阵
    cov_matrix = np.cov(standardized_data, rowvar=True)

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 选择前 n_components 个特征值和对应的特征向量
    top_eigenvalue_indices = np.argsort(eigenvalues)[::-1][:n_components]
    selected_eigenvectors = eigenvectors[:, top_eigenvalue_indices]

    # 将数据投影到选定的特征向量上
    X2 = np.dot(standardized_data.T, selected_eigenvectors)

    return X2

# PCA处理函数1
def perform_pca1(X1, n_components):
    # 仅选择数值列进行均值计算
    numeric_columns = X1.iloc[:, 1:].select_dtypes(include=[float, int])

    # 计算数值列的均值
    mean = np.mean(numeric_columns, axis=1)

    # 使用计算得到的均值对数据进行标准化
    standardized_data = (X1.iloc[:, 1:].T - mean) / np.std(numeric_columns, axis=1, ddof=1)

    if np.any(np.isinf(standardized_data)) or np.any(np.isnan(standardized_data)):
        standardized_data = np.nan_to_num(standardized_data, nan=0.0, posinf=0.0, neginf=0.0)

    # 计算协方差矩阵
    cov_matrix = np.cov(standardized_data, rowvar=True)

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 选择前 n_components 个特征值和对应的特征向量
    top_eigenvalue_indices = np.argsort(eigenvalues)[::-1][:n_components]
    selected_eigenvectors = eigenvectors[:, top_eigenvalue_indices]

    # 将数据投影到选定的特征向量上
    X3 = np.dot(standardized_data.T, selected_eigenvectors)

    return X3


def build_adjacency_matrix(normalized_data, threshold=0, normalize_weights="log_per_cell", edge_norm=True):
    # 如果输入是 pandas.DataFrame，则转换为 numpy.ndarray
    if isinstance(normalized_data, pd.DataFrame):
        normalized_data = normalized_data.to_numpy()

    num_cells, num_genes = normalized_data.shape

    # 非零元素的行索引（细胞）和列索引（基因）
    row_idx, gene_idx = np.nonzero(normalized_data > threshold)

    # 选择不同的标准化方式
    if normalize_weights == "none":
        X1 = normalized_data
    elif normalize_weights == "log_per_cell":
        X1 = np.log1p(normalized_data)
        X1 = X1 / (np.sum(X1, axis=1, keepdims=True) + 1e-6)
    elif normalize_weights == "per_cell":
        X1 = normalized_data / (np.sum(normalized_data, axis=1, keepdims=True) + 1e-6)

    # 选取非零值
    non_zeros = X1[(row_idx, gene_idx)]

    # 计算每个细胞的值的总和（按行求和）
    D_sum = np.sum(X1, axis=1)

    # 初始化细胞（行）和基因（列）的邻接矩阵
    adjacency_matrix = np.zeros((num_cells, num_genes))

    # 填充邻接矩阵，使用加权边
    for r, c, w in zip(row_idx, gene_idx, non_zeros):
        adjacency_matrix[r, c] = w / D_sum[r]

    # 如果 edge_norm 为 True，则归一化边
    if edge_norm:
        for i in range(num_cells):
            row_sum = np.sum(adjacency_matrix[i])
            if row_sum > 0:
                adjacency_matrix[i] = adjacency_matrix[i] / row_sum

    # 将邻接矩阵转换为带有适当标签的 DataFrame
    cell_labels = ['cell_{}'.format(i) for i in range(num_cells)]
    gene_labels = ['gene_{}'.format(i) for i in range(num_genes)]
    adjacency_df = pd.DataFrame(adjacency_matrix, index=cell_labels, columns=gene_labels)

    return adjacency_df

# 内积解码器类
class InnerProductDecoder(nn.Module):
    def __init__(self, activation=torch.sigmoid, dropout=0.1):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.activation = activation

    def forward(self, z):
        z = F.dropout(z, self.dropout)
        adj = self.activation(torch.mm(z, z.t()))
        return adj


class GraphConv(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, norm=None, allow_zero_in_degree=True):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.norm = norm

    def forward(self, graph, features):
        h = torch.matmul(graph, features)
        h = self.linear(h)
        if self.activation:
            h = self.activation(h)
        if self.norm:
            h = self.norm(h)
        return h

class GCNAE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation=None, norm=None, dropout=0.1, hidden=None, hidden_relu=False, hidden_bn=False, agg="sum"):
        super(GCNAE, self).__init__()
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation, norm=norm))
        for _ in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation, norm=norm))
        self.decoder = InnerProductDecoder(activation=lambda x: x)
        self.hidden = hidden
        if hidden is not None:
            enc = []
            for i, _ in enumerate(hidden):
                if i == 0:
                    enc.append(nn.Linear(n_hidden, hidden[i]))
                else:
                    enc.append(nn.Linear(hidden[i-1], hidden[i]))
                if hidden_bn and i != len(hidden):
                    enc.append(nn.BatchNorm1d(hidden[i]))
                if hidden_relu and i != len(hidden):
                    enc.append(nn.LeakyReLU(negative_slope=0.01))
            self.encoder = nn.Sequential(*enc)

    def forward(self, A, features):
        x = features
        for layer in self.layers:
            if self.dropout is not None:
                x = self.dropout(x)
            x = layer(A, x)
        x = x.view(x.shape[0], -1)
        if self.hidden is not None:
            x = self.encoder(x)
        adj_rec = self.decoder(x)
        return adj_rec, x


def plot_clusters(data, labels, title='Clusters', save_path=None, tsne_csv_path=None):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(data)

    # 将 t-SNE 结果保存为 CSV 文件
    if tsne_csv_path:
        tsne_df = pd.DataFrame(tsne_result, columns=["tSNE_1", "tSNE_2"])
        tsne_df["Cluster"] = labels
        tsne_df.to_csv(tsne_csv_path, index=False)

    plt.figure(figsize=(12, 8))

    # 使用较大范围的颜色映射确保每个簇都有唯一的颜色
    num_clusters = len(set(labels))
    colors = sns.color_palette('viridis', num_clusters)

    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, palette=colors, s=50)
    plt.title(title)
    plt.legend(loc='upper right', markerscale=2)  # 设置图例位置和标记大小
    plt.tight_layout()  # 调整布局，防止标签重叠

    # 保存图像到指定路径
    if save_path:
        plt.savefig(save_path)

    plt.show()

# 获取距离质心最近的前30%点的索引
def get_top_30_percent_nearest_centroids(Z_df, cluster_labels_df, n_clusters, top_percent):
    top_30_percent_indices = []

    for cluster in range(n_clusters):
        # 获取当前簇的所有点
        cluster_indices = np.where(cluster_labels_df['Cluster'] == cluster)[0]
        cluster_points = Z_df.iloc[cluster_indices]

        # 计算质心
        centroid = cluster_points.mean(axis=0)

        # 计算每个点到质心的距离
        distances = pairwise_distances(cluster_points, centroid.values.reshape(1, -1), metric='euclidean').flatten()

        # 获取距离质心最近的前30%的点
        top_30_percent_count = int(len(distances) * top_percent)
        top_30_percent_idx = np.argsort(distances)[:top_30_percent_count]
        top_30_percent_indices.extend(cluster_indices[top_30_percent_idx])

    return top_30_percent_indices

# 可视化聚类结果并标记前30%点
def plot_clusters_with_top_30_percent(data, labels, top_indices, title='Clusters', save_path=None, top_color='red'):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(data)
    plt.figure(figsize=(12, 8))
    num_clusters = len(set(labels))
    colors = sns.color_palette('viridis', num_clusters)
    scatter = sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, palette=colors, s=50)
    sns.scatterplot(x=tsne_result[top_indices, 0], y=tsne_result[top_indices, 1], color=top_color, s=50, label='Top 30%')
    scatter.legend(loc='upper right', markerscale=2)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# 定义数据集
class CellDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 定义生成器
class Generator(nn.Module):
    def __init__(self, latent_dim, feature_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, feature_dim),
            nn.Tanh()  # 假设特征值在[-1, 1]
        )

    def forward(self, x):
        return self.model(x)

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, num_classes + 1),  # 增加一个类别用于假样本
        )

    def forward(self, x):
        return self.model(x)

# 训练函数
def train_gan(generator, discriminator, data_loader, optimizer_g, optimizer_d, criterion, num_epochs, latent_dim, num_classes):
    for epoch in range(num_epochs):
        for inputs, labels in data_loader:
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.LongTensor)
            batch_size = inputs.size(0)

            # 训练判别器
            optimizer_d.zero_grad()

            # 真实样本
            real_samples = inputs
            real_outputs = discriminator(real_samples)
            real_loss = criterion(real_outputs[:, :-1], labels)

            # 生成假样本
            noise = torch.randn(batch_size, latent_dim)
            fake_samples = generator(noise)
            fake_outputs = discriminator(fake_samples)

            # 假样本标签设为假样本类别（最后一个类别）
            fake_labels = torch.full((batch_size,), num_classes, dtype=torch.long)
            fake_loss = criterion(fake_outputs, fake_labels)

            # 判别器总损失
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            # 训练生成器
            optimizer_g.zero_grad()
            noise = torch.randn(batch_size, latent_dim)
            fake_samples = generator(noise)
            outputs = discriminator(fake_samples)
            g_loss = criterion(outputs[:, :-1], labels)  # 欺骗判别器的目标是让假样本被认为是真样本
            g_loss.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")
# t-SNE 可视化函数
def visualize_tsne(features, labels, title, save_path):
    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)

    # 创建 DataFrame 包含 t-SNE 降维结果和标签
    tsne_df = pd.DataFrame({
        'tsne_dim1': tsne_results[:, 0],
        'tsne_dim2': tsne_results[:, 1],
        'label': labels
    })

    # 绘制 t-SNE 图
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='tsne_dim1', y='tsne_dim2',
        hue='label', palette='viridis',
        data=tsne_df,
        legend='full',  # 显示完整的图例
        alpha=0.8,       # 设置透明度
        s=50             # 设置点的大小
    )
    plt.title(title)
    plt.savefig(save_path)
    plt.show()

# 强制标签数一致性函数
def enforce_label_consistency(predicted_labels, num_classes):
    unique, counts = np.unique(predicted_labels, return_counts=True)
    label_count = dict(zip(unique, counts))

    # 如果标签数少于预期，重复现有标签
    while len(label_count) < num_classes:
        min_label = min(label_count, key=label_count.get)
        min_count = label_count[min_label]
        for i in range(num_classes):
            if i not in label_count:
                predicted_labels[np.where(predicted_labels == min_label)[0][:min_count]] = i
                label_count[i] = min_count
                break

    # 如果标签数多于预期，合并多余标签
    while len(label_count) > num_classes:
        max_label = max(label_count, key=label_count.get)
        max_count = label_count[max_label]
        for i in range(num_classes):
            if i in label_count and label_count[i] < max_count:
                predicted_labels[np.where(predicted_labels == max_label)[0][:max_count - label_count[i]]] = i
                label_count[i] += max_count - label_count[i]
                del label_count[max_label]
                break

    return predicted_labels

# 主程序
if __name__ == '__main__':
    start_time_1 = time.time()
    #指定CSV文件路径
    expressionFilename = 'E:/jsj/user06/zz/EMATB7678/EMATB7678.csv'
    # 读取CSV文件到DataFrame
    df = pd.read_csv(expressionFilename, index_col=0, header=0)
    # 调用preprocessingCSV函数进行数据预处理
    processed_df = preprocessingCSV(df, expressionFilename)
    # 打印处理后的DataFrame信息
    print(processed_df)
    processed_df.to_csv('E:/jsj/user06/zz/EMATB7678/scGGC/p1.csv', index=True, header=True)

    expressionFilename = 'E:/jsj/user06/zz/EMATB7678/scGGC/p1.csv'
    # 读取CSV文件到DataFrame
    df = pd.read_csv(expressionFilename, index_col=0, header=0)
    print(df)

    numeric_columns = df.select_dtypes(include=[float, int])
    normalized_data = numeric_columns.div(numeric_columns.sum())
    # normalized_data.to_csv('E:/jsj/user06/zz/GSE138852/GSE138852/GSE138852_counts/normalized_df.csv', index=True, header=True)

    X2 = perform_pca(normalized_data, n_components=1300)
    pca_df = pd.DataFrame(X2)
    # pca_df.to_csv('E:/jsj/user06/zz/GSE138852/GSE138852/GSE138852_counts/pca_result.csv', index=False)
    print("X2的形状:", X2.shape)

    X1 = normalized_data.T.dot(X2).astype(float)
    print(X1)
    print("X1的形状:", X1.shape)

    X3 = perform_pca1(X1, n_components=800)
    B = pairwise_distances(X3, metric='euclidean')
    print(B)
    print("B的形状:", B.shape)

    start_time_2 = time.time()

    # 使用KNN算法获取细胞间的邻接图
    knn = NearestNeighbors(n_neighbors=10, metric='precomputed')
    knn.fit(B)# 使用距离矩阵B进行KNN训练
    # 获取K个最近邻居的图表示
    knn_graph = knn.kneighbors_graph()
    # 注意：这将创建一个较大的矩阵，可能占用大量内存
    A1 = knn_graph.toarray()

    end_time_2 = time.time()
    elapsed_time_2 = end_time_2 - start_time_2
    print(f"Saving CSV file took {elapsed_time_2:.2f} seconds.")

    # 如果需要稀疏格式的邻接矩阵以节省内存
    (i, j) = knn_graph.nonzero()
    A1_sparse = coo_matrix((np.ones_like(i), (i, j)), shape=knn_graph.shape)
    # 如果需要将稀疏矩阵转换为普通矩阵格式，可以使用toarray()方法
    A1_full = A1_sparse.toarray()
    print(A1_full)

    # 对邻接矩阵进行行归一化处理
    row_sums = A1_full.sum(axis=1)
    A1_normalized = A1_full / row_sums[:, np.newaxis]
    print(A1_normalized)

    # # 将邻接矩阵 A1_normalized 存储为 CSV 文件
    # pd.DataFrame(A1_normalized).to_csv('E:/jsj/user06/zz/GSE138852/GSE138852/GSE138852_counts/A1_normalized.csv'
    #                                    , index=False, header=False)

    start_time_3 = time.time()
    W = build_adjacency_matrix(normalized_data)
    W = W.T
    # # # 打印邻接矩阵的形状
    # # print("邻接矩阵的形状:", W.shape)
    W.to_csv('E:/jsj/user06/zz/EMATB7678/scGGC/W_original.csv', index=False, header=False)
    end_time_3 = time.time()
    elapsed_time_3 = end_time_3 - start_time_3
    print(f"3 file took {elapsed_time_3:.2f} seconds.")

    # # # 读取原始的邻接矩阵文件
    W_original = pd.read_csv('E:/jsj/user06/zz/EMATB7678/scGGC/W_original.csv', header=None)

    # 对邻接矩阵进行行归一化处理
    row_sums = W_original.sum(axis=1)
    W_normalized = W_original.div(row_sums, axis=0)
    W_normalized.to_csv('E:/jsj/user06/zz/EMATB7678/scGGC/W.csv', index=False, header=False)

    ## 添加权重系数
    alpha = 0.1 # 调整 alpha 的值，使得 A1 和 W 的系数之和为 1
    beta = 0.9 # 调整 beta 的值
    weighted_A1 = alpha * A1_normalized
    weighted_W = beta * W_normalized
    # 确保加权后的 A1 和 W 的系数之和为 1
    sum_of_coefficients = alpha + beta
    weighted_A1 /= sum_of_coefficients
    weighted_W /= sum_of_coefficients

    # 创建零矩阵
    zero_matrix1 = W_normalized.T
    # 创建单位矩阵
    identity_matrix1 = np.eye(2000)
    al = 0.9 # 调整 alpha 的值，使得 A1 和 W 的系数之和为 1
    be = 0.1 # 调整 beta 的值
    zero_matrix = alpha * zero_matrix1
    identity_matrix = beta * identity_matrix1
    sum_of_coefficients = al + be
    zero_matrix /= sum_of_coefficients
    identity_matrix /= sum_of_coefficients


    # Print the shapes of the matrices
    print("Shape of weighted_A1:", weighted_A1.shape)
    print("Shape of weighted_W:", weighted_W.shape)
    print("Shape of zero_matrix:", zero_matrix.shape)
    print("Shape of identity_matrix:", identity_matrix.shape)

    # 合并矩阵
    A = np.block([[weighted_A1, weighted_W], [zero_matrix, identity_matrix]])
    A_tensor = torch.tensor(A, dtype=torch.float32)
    print(A)
    print("A的形状:", A.shape)

    # 将 NumPy 数组转换为 DataFrame
    A_df = pd.DataFrame(A)
    A_df.to_csv('E:/jsj/user06/zz/EMATB7678/scGGC/A_matrix.csv', index=False)

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
    train_model(encoder_decoder_model, A_tensor, X_combined_tensor, labels, optimizer, loss_fn, num_epochs=100)
    end_time_4 = time.time()
    elapsed_time_4 = end_time_4 - start_time_4
    print(f"3 file took {elapsed_time_4:.2f} seconds.")

    _, embeddings = encoder_decoder_model(A_tensor, X_combined_tensor)
    latent_variables = embeddings.detach().numpy()
    latent_variables_df = pd.DataFrame(latent_variables)
    latent_variables_df.to_csv('E:/jsj/user06/zz/EMATB7678/scGGC/latent_variables.csv', index=False, header=False)
    latent_variables = latent_variables[:6666]

    cell_names = df.columns.tolist()
    Z_df = pd.DataFrame(latent_variables, index=cell_names)
    Z_df.to_csv('E:/jsj/user06/zz/EMATB7678/scGGC/Z_df.csv', header=False)

    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(Z_df)
    cluster_labels = kmeans.labels_
    cluster_labels_df = pd.DataFrame(cluster_labels, index=cell_names, columns=['Cluster'])
    cluster_labels_df.index.name = "CellName"
    cluster_labels_df.to_csv('E:/jsj/user06/zz/EMATB7678/scGGC/scGGCn1.csv')

    # 读取真实标签
    true_labels_filename = 'E:/jsj/user06/zz/EMATB7678/clusters.csv'
    true_labels_df = pd.read_csv(true_labels_filename, index_col=0)
    true_labels = true_labels_df['Cluster8'].values

    # 计算并打印ARI
    ari = adjusted_rand_score(true_labels, cluster_labels)
    print(f'Adjusted Rand Index (ARI): {ari}')


    # # # 调用修改后的 plot_clusters 函数，并保存图像
    tsne_plot_path = 'E:/jsj/user06/zz/EMATB7678/scGGC/scGGC66.png'
    # plot_clusters(Z_df.values, cluster_labels, title='t-SNE Visualization of Cells', save_path=tsne_plot_path)
    # 调用 plot_clusters 函数并保存 t-SNE 数据到 CSV 文件
    tsne_csv_path = 'E:/jsj/user06/zz/EMATB7678/scGGC/GraphsctSNE_results.csv'
    plot_clusters(Z_df.values, cluster_labels, title='t-SNE Visualization of Cells', save_path=tsne_plot_path,
                  tsne_csv_path=tsne_csv_path)
    # 获取前30%距离质心最近的点的索引
    top_30_percent_indices = get_top_30_percent_nearest_centroids(Z_df, cluster_labels_df, n_clusters=8, top_percent=0.5)

    # 提取这些点的细胞名和标签
    top_30_percent_df = cluster_labels_df.iloc[top_30_percent_indices]
    top_30_percent_df.to_csv('E:/jsj/user06/zz/EMATB7678/scGGC/selected_cells.csv', index=True, header=True)


    # tsne_plot_path1 = 'E:/jsj/user06/zz/10x_data1/scGGC/splot.png'
    # plot_clusters_with_top_30_percent(Z_df.values, cluster_labels, top_30_percent_indices, title='t-SNE Visualization of Selected Cells', save_path=tsne_plot_path1, top_color='red')

    # 加载选择的细胞
    selected_cells_path = 'E:/jsj/user06/zz/EMATB7678/scGGC/selected_cells.csv'
    selected_cells_df = pd.read_csv(selected_cells_path)

    # 获取选择的细胞的索引
    selected_cell_indices = selected_cells_df['CellName'].tolist()

    # 加载全部细胞数据
    gene_expression_path = 'E:/jsj/user06/zz/EMATB7678/scGGC/p1.csv'
    gene_expression_data = pd.read_csv(gene_expression_path, index_col=0)

    # 加载聚类标签
    cluster_labels_path = 'E:/jsj/user06/zz/EMATB7678/scGGC/scGGCn1.csv'
    cluster_labels_df = pd.read_csv(cluster_labels_path)
    cluster_labels = cluster_labels_df.set_index('CellName')['Cluster'].to_dict()

    # 获取标签的数量
    num_classes = len(set(cluster_labels.values()))

    # 从全部细胞数据中筛选出选定的细胞数据
    selected_gene_expression_data = gene_expression_data[selected_cell_indices]

    # 数据预处理
    scaler = StandardScaler()
    selected_gene_expression_data_scaled = scaler.fit_transform(selected_gene_expression_data.T)

    # 转换为 PyTorch Tensor 并创建 dataset
    features_tensor = torch.tensor(selected_gene_expression_data_scaled).float()
    selected_labels = [cluster_labels[cell] for cell in selected_cell_indices]
    labels_tensor = torch.tensor(selected_labels).long()
    cell_dataset = CellDataset(features_tensor, labels_tensor)

    # 数据加载器
    batch_size = 64
    train_loader = DataLoader(cell_dataset, batch_size=batch_size, shuffle=True)

    # 超参数
    latent_dim = 100
    feature_dim = features_tensor.shape[1]

    # 初始化模型
    generator = Generator(latent_dim, feature_dim)
    discriminator = Discriminator(feature_dim, num_classes)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # 训练GAN
    num_epochs = 30
    train_gan(generator, discriminator, train_loader, optimizer_g, optimizer_d, criterion, num_epochs, latent_dim,
              num_classes)

    # 使用训练好的对抗模型对所有细胞进行分类
    all_gene_expression_data_scaled = scaler.transform(gene_expression_data.T)
    all_features_tensor = torch.tensor(all_gene_expression_data_scaled).float()

    # 使用判别器对所有细胞进行分类
    with torch.no_grad():
        all_outputs = discriminator(all_features_tensor)
        predicted_labels = torch.argmax(all_outputs[:, :-1], dim=1).numpy()  # 只考虑真实类别

    # 强制标签数一致
    enforced_labels = enforce_label_consistency(predicted_labels, num_classes)
    enforced_labels += 1
    # 创建包含细胞名及其重新分类标签的DataFrame
    reclassified_cells_df = pd.DataFrame({
        'CellName': gene_expression_data.columns,
        'ReclassifiedLabel': enforced_labels
    })

    # 保存到CSV文件
    reclassified_cells_df.to_csv('E:/jsj/user06/zz/EMATB7678/scGGC/reclassified_all.csv', index=False)

    # 读取真实标签
    true_labels_path = 'E:/jsj/user06/zz/EMATB7678/clusters.csv'  # 替换为真实标签文件的路径
    true_labels_df = pd.read_csv(true_labels_path, index_col=0)
    true_labels = true_labels_df['Cluster8'].values

    # 计算并打印ARI
    ari = adjusted_rand_score(true_labels, enforced_labels)
    print(f'Adjusted Rand Index (ARI): {ari}')

    # 可视化t-SNE结果并保存图像
    tsne_plot_path1 = 'E:/jsj/user06/zz/EMATB7678/scGGC/gan4.png'
    visualize_tsne(all_gene_expression_data_scaled, enforced_labels, 't-SNE Visualization of Reclassified Cells',
                   tsne_plot_path1)


    end_time_1 = time.time()
    elapsed_time_1 = end_time_1 - start_time_1
    print(f"Saving CSV file took {elapsed_time_1:.2f} seconds.")
