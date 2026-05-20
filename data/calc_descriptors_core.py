import pandas as pd
from typing import List
from tqdm import tqdm
from rdkit.Chem import Descriptors


def remove_highly_correlated_features(df: pd.DataFrame, threshold: float = 0.95) -> pd.Index:
    """Remove features above the correlation threshold, returning retained feature names."""
    df_corr = df.corr()
    df_corr = abs(df_corr)
    columns = df_corr.columns

    # Set diagonal values to zero
    for i in range(0, len(columns)):
        df_corr.iloc[i, i] = 0

    while True:
        df_max_column_value = df_corr.max()
        max_corr = df_max_column_value.max()
        query_column = df_max_column_value.idxmax()
        target_column = df_corr[query_column].idxmax()

        if max_corr < threshold:
            break

        # Remove the feature that has higher total correlation with other features
        if sum(df_corr[query_column]) <= sum(df_corr[target_column]):
            delete_column = target_column
        else:
            delete_column = query_column

        df_corr.drop([delete_column], axis=0, inplace=True)
        df_corr.drop([delete_column], axis=1, inplace=True)

    return df_corr.columns


def main_calculate_descriptors(df: pd.DataFrame, discrete_columns: List[str], label_columns: List[str], corr_threshold: float = 0.95) -> pd.DataFrame:
    """Calculate molecular descriptors, remove correlated features, and return with label columns."""

    label_df = df[label_columns].reset_index(drop=True)

    # Select relevant columns
    df = df[["NAME", 'inchikey', 'smiles', 'ROMol']].reset_index(drop=True)

    # Calculate molecular descriptors
    for i, j in tqdm(Descriptors.descList):
        df[i] = df["ROMol"].map(j)

    x1_discrete = df[discrete_columns]

    # Remove rows with missing values
    autoscaled_x1 = x1_discrete.dropna(how="any", axis=1)

    # Standardize features
    autoscaled_x1_r = (autoscaled_x1 - autoscaled_x1.mean()) / autoscaled_x1.std()

    # Remove highly correlated features
    x_corr = list(remove_highly_correlated_features(autoscaled_x1_r, corr_threshold))
    x1_done_corr = autoscaled_x1_r[x_corr]

    # Combine original data, label columns, and selected descriptor features
    df_con_tr = pd.concat([df, label_df, x1_done_corr], axis=1)

    return df_con_tr
