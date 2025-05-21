import numpy as np
import pandas as pd
from rdkit.Chem import AllChem


def generate_morgan_fingerprints(df: pd.DataFrame, radius: int, n_bits: int) -> np.ndarray:
    """
    Generate Morgan fingerprints for molecules in the dataframe.
    
    Args:
        df: DataFrame containing RDKit molecule objects in a column named 'ROMol'
        radius: The radius of the Morgan fingerprint. Higher values capture more extended 
                connectivity information around each atom
        n_bits: The length of the bit vector. Larger values reduce the chance of bit collisions
    
    Returns:
        A numpy array containing fingerprints, where each row represents a molecule 
        and each column represents a bit in the fingerprint
    """
    fingerprints = []
    for i, mol in enumerate(df["ROMol"]):
        try:
            fingerprint = [j for j in AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)]
            fingerprints.append(fingerprint)
        except:
            print(f"Error processing molecule at index {i}")
            continue
    fingerprints = np.array(fingerprints)
    return fingerprints

def ECFP4096bit(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    categories: List[str], 
    lightgbm_model: lgb.LGBMClassifier
) -> Dict[str, Dict[str, float]]:
    """Generate Morgan fingerprints and evaluate LightGBM models for various categories
    
    Args:
        train_df: Training DataFrame containing molecular data
        test_df: Test DataFrame containing molecular data
        categories: List of category names to evaluate
        lightgbm_model: Configured LightGBM classifier instance
    
    Returns:
        Dictionary mapping categories to their training and test scores
    """
    # Generate Morgan fingerprints
    train_desc = np.array(generate_morgan_fingerprints(train_df,3,4096))
    test_desc = np.array(generate_morgan_fingerprints(test_df,3,4096))
    
    # Evaluate each category
    results = {}
    for category in categories:
        results[category] = train_and_evaluate_model(
            train_df, test_df, train_desc, test_desc, category, lightgbm_model
        )
    
    return results
