import pandas as pd
import numpy as np

def preprocessing_csv(df, expression_filename=None):
    if isinstance(df, pd.DataFrame):
        data = df
    else:
        data = pd.read_csv(expression_filename, index_col=0, header=0)

    data = data[data[data.columns[1:]].astype('bool').mean(axis=1) >= 0.01]
    print('After preprocessing, {} genes remaining'.format(data.shape[0] - 1))

    data = data.loc[(data.iloc[1:, 1:].var(axis=1, numeric_only=True).sort_values()[-2000:]).index]
    data.fillna(0, inplace=True)

    return data

def perform_pca(normalized_data, n_components):
    numeric_columns = normalized_data.iloc[:, 1:].select_dtypes(include=[float, int])
    mean = np.mean(numeric_columns, axis=1)
    standardized_data = (normalized_data.iloc[:, 1:].T - mean) / np.std(numeric_columns, axis=1, ddof=1)

    if np.any(np.isinf(standardized_data)) or np.any(np.isnan(standardized_data)):
        standardized_data = np.nan_to_num(standardized_data, nan=0.0, posinf=0.0, neginf=0.0)

    cov_matrix = np.cov(standardized_data, rowvar=True)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    top_eigenvalue_indices = np.argsort(eigenvalues)[::-1][:n_components]
    selected_eigenvectors = eigenvectors[:, top_eigenvalue_indices]
    X2 = np.dot(standardized_data.T, selected_eigenvectors)

    return X2
