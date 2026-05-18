import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def export_upper_triangle_matrix(similarity_matrix, filename):
    n = len(similarity_matrix)
    upper_matrix = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(i, n):
            upper_matrix[i, j] = similarity_matrix[i, j]
    
    df = pd.DataFrame(
        upper_matrix,
        index=[f"cmp_{i}" for i in range(n)],
        columns=[f"cmp_{i}" for i in range(n)]
    )
    
    df.to_excel(filename, index=False, float_format='%.6f')
    return df

def frobenius_distance(A, B):
    A = np.array(A)
    B = np.array(B)
    mask = np.triu(np.ones_like(A, dtype=bool))
    A_upper = A[mask]
    B_upper = B[mask]
    return np.linalg.norm(A_upper - B_upper)

def pearson_correlation(A, B):
    A = np.array(A)
    B = np.array(B)
    mask = np.triu(np.ones_like(A, dtype=bool))
    A_upper = A[mask]
    B_upper = B[mask]
    return pearsonr(A_upper, B_upper)[0]

def spearman_correlation(A, B):
    A = np.array(A)
    B = np.array(B)
    mask = np.triu(np.ones_like(A, dtype=bool))
    A_upper = A[mask]
    B_upper = B[mask]
    return spearmanr(A_upper, B_upper)[0]

def compare_matrices(a, b, c, d):
    matrices = [a, b, c, d]
    n = 4
    
    frobenius_matrix = np.full((n, n), np.nan)
    pearson_matrix = np.full((n, n), np.nan)
    spearman_matrix = np.full((n, n), np.nan)
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                frobenius_matrix[i, j] = 0.0
                pearson_matrix[i, j] = 1.0
                spearman_matrix[i, j] = 1.0
            else:
                frobenius_matrix[i, j] = frobenius_distance(matrices[i], matrices[j])
                pearson_matrix[i, j] = pearson_correlation(matrices[i], matrices[j])
                spearman_matrix[i, j] = spearman_correlation(matrices[i], matrices[j])
    
    return frobenius_matrix, pearson_matrix, spearman_matrix
