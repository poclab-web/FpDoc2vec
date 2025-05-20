import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from typing import Dict, List, Optional, Tuple, Union, Any
from rdkit import Chem
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


def train_lightgbm_model(fingerprints: np.ndarray, target_values: np.ndarray) -> lgb.LGBMClassifier:
    """
    Train a LightGBM model with optimized hyperparameters
    
    Args:
        fingerprints: numpy array of molecular fingerprints
        target_values: numpy array of target values (0 or 1)
        
    Returns:
        trained LightGBM model
    """
    
    model = lgb.LGBMClassifier(**params)
    model.fit(fingerprints, target_values)
    return model


def calculate_shap_values(model: lgb.LGBMClassifier, 
                          features: np.ndarray, 
                          feature_perturbation: str = 'tree_path_dependent', 
                          model_output: str = 'raw') -> np.ndarray:
    """
    Calculate SHAP values for the trained model
    
    Args:
        model: trained LightGBM model
        features: feature matrix used for explanation
        feature_perturbation: method used to perturb features for SHAP calculation
        model_output: type of model output to explain
        
    Returns:
        SHAP values array
    """
    explainer = shap.TreeExplainer(
        model=model,
        feature_perturbation=feature_perturbation,
        model_output=model_output
    )
    
    return explainer.shap_values(features)


def main(input_path: str, purpose: str, output_path: str) -> None:
    """
    Main function to run the SHAP analysis pipeline
    
    Args:
        input_path: Path to the pickled DataFrame with molecule data
        purpose: Target biological role to analyze
        output_path: Path to save the SHAP values output
        
    Returns:
        None
    """
    # Load dataset
    with open(input_path, "rb") as f:
        df = pickle.load(f)
    
    # Specify the target biological role here
    y = np.array([1 if i == purpose else 0 for i in df[purpose]])
    
    # Generate molecular fingerprints
    fingerprints = generate_morgan_fingerprints(df, 3, 4096)
    
    # Train the model
    model = train_lightgbm_model(fingerprints, y)
    
    # Calculate SHAP values
    shap_values = calculate_shap_values(model, fingerprints)
    
    # Save SHAP values
    with open(output_path, 'wb') as f:
        pickle.dump(shap_values, f)


if __name__ == "__main__":
    # Note: params should be defined before this function or passed as an argument
    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'random_state': 42
    }  # Example params - replace with your actual params
    # Example usage
    main(input_path="path/to/molecule_data.pkl", 
         purpose="antioxidant", 
         output_path="path/to/shap_results.pkl")
