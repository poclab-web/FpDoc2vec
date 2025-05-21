import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Mapping
import pandas as pd
from rdkit.Chem import MACCSkeys
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


def generate_maccs_fingerprints(df: pd.DataFrame) -> Tuple[List[Optional[List[int]]], List[int]]:
    """
    Generate MACCS fingerprints for molecules in the dataframe
    
    Args:
        df: DataFrame containing RDKit molecule objects in the 'ROMol' column
        
    Returns:
        Tuple containing:
        - List of fingerprints (lists of on-bit indices) or None for invalid molecules
        - List of indices of invalid molecules in the original dataframe
    """
    maccs_features = []
    invalid_indices = []
    
    for idx, mol in enumerate(df["ROMol"]):
        fps = MACCSkeys.GenMACCSKeys(mol)
        fp_bits = list(fps.GetOnBits())
        
        if len(fp_bits) == 0:
            print(f"Invalid MACCS for index {idx}")
            maccs_features.append(None)
            invalid_indices.append(idx)
        else:
            maccs_features.append(fp_bits)
            
    return maccs_features, invalid_indices


def create_index_mapping(df_length: int, invalid_indices: List[int]) -> Dict[int, int]:
    """
    Create a mapping from original dataframe indices to filtered dataframe indices
    
    Args:
        df_length: Length of the original dataframe
        invalid_indices: List of indices to exclude from the mapping
        
    Returns:
        Dictionary mapping original indices to new filtered indices
    """
    original_to_filtered = {}
    filtered_idx = 0
    
    for orig_idx in range(df_length):
        if orig_idx not in invalid_indices:
            original_to_filtered[orig_idx] = filtered_idx
            filtered_idx += 1
            
    return original_to_filtered


def evaluate_with_keys(
    lightgbm_model: lgb.LGBMClassifier,
    df: pd.DataFrame,
    df_keys: pd.DataFrame,
    X_vec_keys: np.ndarray,
    categories: List[str],
    index_mapping: Dict[int, int]
) -> Dict[str, Union[List[float], float]]:
    """
    Evaluate model performance using Doc2Vec vectors
    
    Args:
        lightgbm_model: Trained LightGBM classifier model
        df: Original dataframe with all molecules
        df_keys: Filtered dataframe with valid molecules
        X_vec_keys: Feature vectors for the filtered molecules
        categories: List of category names to evaluate
        index_mapping: Mapping from original dataframe indices to filtered indices
        
    Returns:
        Dictionary containing evaluation metrics:
        - 'train_scores': List of F1 scores on training data
        - 'test_scores': List of F1 scores on test data
        - 'mean_train': Mean F1 score on training data
        - 'mean_test': Mean F1 score on test data
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    
    results = {}
    for category_idx, category in enumerate(categories):
        keys_train_f1_list, keys_test_f1_list = [], []
        
        # Prepare target variables
        y = np.array([1 if i == category else 0 for i in df[category]])
        y_keys = np.array([1 if i == category else 0 for i in df_keys[category]])
        
        for train_idx, test_idx in skf.split(range(len(df)), y):
            # Convert indices to MACCS-compatible indices
            train_idx_keys = [index_mapping[idx] for idx in train_idx if idx in index_mapping]
            test_idx_keys = [index_mapping[idx] for idx in test_idx if idx in index_mapping]
            
            # Only proceed if we have enough samples in both train and test sets
            if len(train_idx_keys) > 0 and len(test_idx_keys) > 0: 
                # Extract training and testing data
                X_train_keys = X_vec_keys[train_idx_keys]
                X_test_keys = X_vec_keys[test_idx_keys]
                y_train_keys = y_keys[train_idx_keys]
                y_test_keys = y_keys[test_idx_keys]
                
                # Train model and make predictions
                lightgbm_model.fit(X_train_keys, y_train_keys)
                y_pred_train_keys = lightgbm_model.predict(X_train_keys)
                y_pred_test_keys = lightgbm_model.predict(X_test_keys)
                
                # Calculate F1 score
                keys_train_f1_list.append(f1_score(y_train_keys, y_pred_train_keys))
                keys_test_f1_list.append(f1_score(y_test_keys, y_pred_test_keys))
        
        print(f"Training Data: {np.mean(keys_train_f1_list)}")
        print(f"Test Data: {np.mean(keys_test_f1_list)}")
    
    return {
        'train_scores': keys_train_f1_list,
        'test_scores': keys_test_f1_list,
        'mean_train': np.mean(keys_train_f1_list),
        'mean_test': np.mean(keys_test_f1_list)
    }

def main_maccs(input_file: str, params: Dict[str, Any], lightgbm_model: lgb.LGBMClassifier, 
               purpose_description: str = "description_remove_stop_words") -> Dict[str, Dict[str, float]]:
    """
    Train and evaluate models using MACCS key fingerprints.
    
    Args:
        input_file: Path to the pickle file containing compound data
        params: Parameters for the Doc2Vec model
        lightgbm_model: Pre-configured LightGBM classifier
        purpose_description: Column name in the DataFrame containing text descriptions
        
    Returns:
        Dictionary mapping category names to evaluation results
    """
    # Load dataset
    with open(input_file, "rb") as f:
        df = pickle.load(f)
    # Generate MACCS fingerprints
    maccs_features, invalid_indices = generate_maccs_fingerprints(df)
    # Create a filtered dataframe with valid MACCS fingerprints
    df_maccs = df.copy()
    df_maccs["maccs"] = maccs_features
    df_maccs = df_maccs.dropna(subset=['maccs'])
    print(f"Number of compounds with valid MACCS keys: {len(df_maccs)}")
    
    # Prepare data for Doc2Vec
    maccs_list = list(df_maccs["maccs"])
    corpus = [sum(doc, []) for doc in df_maccs[purpose_description]]
    
    # Define categories
    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin',
        'flavouring_agent', 'agrochemical', 'volatile_oil', 'antibacterial_agent', 'insecticide'
    ]
    
    # Build Doc2Vec model
    model = build_doc2vec_model(corpus, maccs_list, params)
    
    # Generate compound vectors
    compound_vec = add_vectors(maccs_list, model)
    X_vec_maccs = np.array([compound_vec[i] for i in range(len(df_maccs))])
    
    # Create index mapping
    index_mapping = create_index_mapping(len(df), invalid_indices)
    
    # Evaluate model performance
    results = evaluate_with_keys(lightgbm_model, df, df_maccs, X_vec_maccs, categories, index_mapping)

    return results
