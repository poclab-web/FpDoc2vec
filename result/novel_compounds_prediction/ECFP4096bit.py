import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Tuple, Any, Optional, Union
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import f1_score
from rdkit.Chem import AllChem
from FpDoc2Vec import load_data, train_and_evaluate_model


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


def main(input_path: str, params: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Load data, generate molecular fingerprints, and evaluate LightGBM models for various categories.
    
    Args:
        input_path: Path to the pickle file containing the input DataFrame
        params: Dictionary of parameters for the LightGBM classifier
    
    Returns:
        Dictionary with evaluation results for each category, where keys are category names
        and values are dictionaries containing performance metrics
    """
    # Load dataset
    with open(input_path, "rb") as f:
        df = pickle.load(f)
    train_df, test_df = load_data()
        
    train_desc = np.array(generate_morgan_fingerprints(train_df))
    test_desc = np.array(generate_morgan_fingerprints(test_df))
    
    # Define categories to evaluate
    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin', 
        'flavouring_agent', 'agrochemical', 'volatile_oil', 'antibacterial_agent', 'insecticide'
    ]
    
    # Create classifier
    lightgbm_model = lgb.LGBMClassifier(**params)
    
    # Evaluate each category
    results = {}
    for category in categories:
        results[category] = train_and_evaluate_model(train_df, test_df, train_desc, test_desc, category, lightgbm_model)
    
    return results


if __name__ == "__main__":
    # Example parameters for the LightGBM classifier
    default_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'random_state': 42
    }
    
    results = main(input_path="path/to/your/data.pkl", params=default_params)
