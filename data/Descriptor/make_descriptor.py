import pickle
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from tqdm import tqdm
from rdkit.Chem import Descriptors
import numpy as np


def remove_highly_correlated_features(df: pd.DataFrame, threshold: float = 0.95) -> pd.Index:
    """
    Remove features with correlation coefficient above the threshold
    
    Parameters:
    -----------
    df : pandas.DataFrame
    threshold : float
        Correlation threshold for feature removal (default: 0.95)
        
    Returns:
    --------
    pandas.Index
        Names of retained features after correlation removal
    """
    df_corr = df.corr()
    df_corr = abs(df_corr)
    columns = df_corr.columns
    
    # Set diagonal values to zero
    for i in range(0, len(columns)):
        df_corr.iloc[i, i] = 0
    
    deleted_features = []
    retained_features = []
    correlation_scores = []
    
    while True:
        columns = df_corr.columns
        max_corr = 0.0
        query_column = None
        target_column = None
        
        df_max_column_value = df_corr.max()
        max_corr = df_max_column_value.max()
        query_column = df_max_column_value.idxmax()
        target_column = df_corr[query_column].idxmax()
        
        if max_corr < threshold:
            # No more correlations above threshold
            break
        else:
            # Found correlation above threshold
            delete_column = None
            saved_column = None
            
            # Remove the feature that has higher correlation with other features
            if sum(df_corr[query_column]) <= sum(df_corr[target_column]):
                delete_column = target_column
                saved_column = query_column
            else:
                delete_column = query_column
                saved_column = target_column
                
            deleted_features.append(delete_column)
            retained_features.append(saved_column)
            correlation_scores.append(max_corr)
            print(delete_column + " " + saved_column)
            
            # Remove the feature from correlation matrix
            df_corr.drop([delete_column], axis=0, inplace=True)
            df_corr.drop([delete_column], axis=1, inplace=True)
    
    return df_corr.columns


def main(input_file: str, discrete_columns: List[str], output_file: str) -> None:
    """
    Process molecular data by calculating descriptors and removing highly correlated features
    
    Parameters:
    -----------
    input_file : str
        Path to a pickle file containing a pandas DataFrame with molecule data
    discrete_columns : List[str]
        List of molecular descriptor column names to analyze
    output_file : str
        Path to save the output pickle file with processed data
        
    Returns:
    --------
    None
    """
    # Load data
    with open(input_file, "rb") as f:
        df = pickle.load(f)
    
    # Process molecule names
    df["NAME"] = [df.iat[i, 0][0] for i in range(len(df))]
    
    # Select relevant columns
    all_columns = ["NAME", 'inchikey', 'smiles', 'ROMol', 'antioxidant',
                  'anti_inflammatory_agent', 'allergen', 'dye', 'toxin', 'flavouring_agent',
                  'agrochemical', 'volatile_oil', 'antibacterial_agent', 'insecticide']
    df = df[["NAME", 'inchikey', 'smiles', 'ROMol']]
    
    # Calculate molecular descriptors
    for i, j in tqdm(Descriptors.descList):
        df[i] = df["ROMol"].map(j)
    
    
    x1_discrete = df[discrete_columns]
    
    # Remove rows with missing values
    autoscaled_x1 = x1_discrete.dropna(how="any", axis=1)
    
    # Standardize features
    autoscaled_x1_r = (autoscaled_x1 - autoscaled_x1.mean()) / autoscaled_x1.std()
        
    # Remove highly correlated features
    x_corr = list(remove_highly_correlated_features(autoscaled_x1_r, 0.95))
    x1_done_corr = autoscaled_x1_r[x_corr]
    
    # Combine original data with selected features
    df_con_tr = pd.concat([df, x1_done_corr], axis=1)
    
    # Save results
    with open(output_file, "wb") as f:
        pickle.dump(df_con_tr, f)


if __name__ == "__main__":
    # Define continuous descriptor columns for analysis
    discrete_columns: List[str] = ['MaxEStateIndex', 'MinEStateIndex', 'qed', 'MolWt', 'MaxPartialCharge', 
                        'MinPartialCharge', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 
                        'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 
                        'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'BalabanJ', 'BertzCT', 'Chi0', 
                        'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 
                        'Chi4n', 'Chi4v', 'HallKierAlpha', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 
                        'PEOE_VSA1', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'SMR_VSA1', 'SMR_VSA10', 
                        'SMR_VSA2', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 
                        'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 
                        'FractionCSP3', 'MolLogP', 'MolMR']
    # Example usage - replace with your actual file paths
    input_file: str = "10genre_dataset.pkl"
    output_file: str = "10genre_descriptor"
    main(input_file, discrete_columns, output_file)
