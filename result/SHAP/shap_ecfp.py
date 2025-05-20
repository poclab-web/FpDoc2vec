import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Dict, List, Optional, Tuple, Union, Any


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


def create_lightgbm_classifier(params: Dict[str, Any]) -> lgb.LGBMClassifier:
    """
    Create and configure LightGBM classifier with optimized parameters
    
    Args:
        params: Dictionary of parameters for LightGBM classifier configuration
        
    Returns:
        Configured LightGBM classifier model (not yet trained)
    """
    
    model = lgb.LGBMClassifier(**params)
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


def main(input_path: str, purpose: str, model: lgb.LGBMClassifier, output_path: str) -> None:
    """
    Main function to run the SHAP analysis pipeline
    
    Args:
        input_path: Path to the pickled DataFrame with molecule data
        purpose: Target biological role to analyze
        model: Pre-configured LightGBM model
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
    model.fit(fingerprints, y)
    
    # Calculate SHAP values
    shap_values = calculate_shap_values(model, fingerprints)
    
    # Save SHAP values
    with open(output_path, 'wb') as f:
        pickle.dump(shap_values, f)


if __name__ == "__main__":
    # Define model parameters
    # Example params - replace with your actual params
    gbm_params: Dict[str, Any] = {
        "boosting_type": "dart", 
        "n_estimators": 444, 
        "learning_rate": 0.07284380689492893, 
        "max_depth": 6, 
        "num_leaves": 41, 
        "min_child_samples": 21, 
        "class_weight": "balanced", 
        "reg_alpha": 1.4922729949843299, 
        "reg_lambda": 2.8809246344115778, 
        "colsample_bytree": 0.5789063337359206, 
        "subsample": 0.5230422589468584, 
        "subsample_freq": 2, 
        "drop_rate": 0.1675163179873052, 
        "skip_drop": 0.49103811434109507, 
        "objective": 'binary', 
        "random_state": 50
    }

    model = create_lightgbm_classifier(params)
    # Example usage - replace with your actual file paths
    input_path = "10genre_dataset.pkl"
    # Please modify according to the purpose.
    purpose="antioxidant"
    output_path = "shap_ecfp_value.pkl"
    target_molecule = "quercetin"
    
    main(input_path, purpose="antioxidant", model, output_path)
