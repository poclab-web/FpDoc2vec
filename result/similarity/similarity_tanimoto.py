import pickle
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem
import pandas as pd

def calculate_tanimoto_similarities(df, target_compound):
    """
    Calculate Tanimoto similarities between a target compound and all compounds.
    
    Args:
        df: DataFrame containing compound information
        target_compound: Name of the target compound
        
    Returns:
        List of Tanimoto similarity scores
    """
    # Generate Morgan fingerprints for all compounds
    morgan_fps = [AllChem.GetMorganFingerprintAsBitVect(df.iat[i, 10], 3, 4096) 
                 for i in range(len(df))]
    
    # Find the index of the target compound
    target_idx = None
    for i in range(len(df)):
        if df.iat[i, 0][0] == target_compound:
            target_idx = i
            break
    
    if target_idx is None:
        raise ValueError(f"Compound '{target_compound}' not found in the dataset")
    
    # Calculate Tanimoto similarities
    similarities = DataStructs.BulkTanimotoSimilarity(morgan_fps[target_idx], morgan_fps)
    
    return similarities

def get_top_similar_compounds(df, similarity_scores, target_compound, n=10):
    """
    Find the top N compounds most similar to the target compound.
    
    Args:
        df: DataFrame containing compound information
        similarity_scores: List of similarity scores
        target_compound: Name of the target compound
        n: Number of similar compounds to return (default: 10)
        
    Returns:
        DataFrame with the top N most similar compounds
    """
    # Add similarity scores to DataFrame
    df_copy = df.copy()
    df_copy["tanimoto"] = similarity_scores
    
    # Sort by similarity score in descending order
    sorted_df = df_copy.sort_values("tanimoto", ascending=False)
    
    # Return the top N compounds (excluding the target compound itself)
    # The first one is usually the compound itself (similarity=1.0)
    return sorted_df.iloc[1:n+1][["NAME", "tanimoto"]]

def main():
    """Load data and calculate Tanimoto similarities to sucrose."""
    # Load dataset
    with open("../../data/10genre_dataset.pkl", "rb") as f:
        df = pickle.load(f)
    
    # Add compound names as a separate column if not already present
    if "NAME" not in df.columns:
        df["NAME"] = [df.iat[i, 0][0] for i in range(len(df))]
    
    # Calculate Tanimoto similarities to sucrose
    target_compound = "sucrose"
    tanimoto_similarities = calculate_tanimoto_similarities(df, target_compound)
    
    # Get top 10 similar compounds
    top_similar = get_top_similar_compounds(df, tanimoto_similarities, target_compound, 10)
    
    # Display results in the same format as before
    print(f"・Similar terms to '{target_compound}' with similarity scores:")
    for idx, row in top_similar.iterrows():
        compound_name = row['NAME']
        similarity_score = row["tanimoto"]
        print(f"  {compound_name}: {similarity_score:.4f}")

if __name__ == "__main__":
    main()
