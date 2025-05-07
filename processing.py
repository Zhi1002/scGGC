import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

def preprocessingCSV(df):
    # 把所有细胞中，表达比例还不到1%的基因去除掉。
    data = df[df[df.columns[1:]].astype('bool').mean(axis=1) >= 0.01]
    print('After preprocessing, {} genes remaining'.format(data.shape[0]))

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