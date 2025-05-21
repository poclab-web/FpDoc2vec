import pickle
import numpy as np
from typing import Dict, List, Tuple, Union, Any
import pandas as pd
from rdkit.Chem import AllChem
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


def add_vectors(fp_list: List[List[int]], model: Doc2Vec) -> List[np.ndarray]:
    """Combine document vectors based on fingerprints
    
    Args:
        fp_list: List of fingerprint lists, where each fingerprint is represented as a list of indices
        model: Trained Doc2Vec model containing document vectors
        
    Returns:
        List of compound vectors as numpy arrays
    """
    compound_vec = []
    for i in fp_list:
        fingerprint_vec = 0
        for j in i:
            fingerprint_vec += model.dv.vectors[j]
        compound_vec.append(fingerprint_vec)
    return compound_vec


def evaluate_category(category: str, 
                      X_vec: np.ndarray, 
                      y: np.ndarray, 
                      lightgbm_model: lgb.LGBMClassifier) -> Dict[str, Union[List[float], float]]:
    """Evaluate model performance for a specific category using cross-validation
    
    Args:
        category: Name of the category being evaluated
        X_vec: Feature matrix as numpy array containing compound vectors
        y: Target array containing binary labels for the category
        lightgbm_model: Pre-configured LightGBM classifier model
        
    Returns:
        Dictionary containing training and test scores:
            - train_scores: List of F1 scores for each fold (training data)
            - test_scores: List of F1 scores for each fold (test data)
            - mean_train: Mean F1 score across all folds (training data)
            - mean_test: Mean F1 score across all folds (test data)
    """
    print(f"## {category} ##")
    test_scores = []
    train_scores = []
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_idx, test_idx in skf.split(range(len(y)), y):
        X_train_vec, X_test_vec = X_vec[train_idx], X_vec[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        lightgbm_model.fit(X_train_vec, y_train)
        y_train_pred = lightgbm_model.predict(X_train_vec)
        y_test_pred = lightgbm_model.predict(X_test_vec)
        
        train_scores.append(f1_score(y_train, y_train_pred))
        test_scores.append(f1_score(y_test, y_test_pred))
    
    print(f"Training Data: {np.mean(train_scores)}")
    print(f"Test Data: {np.mean(test_scores)}")
    
    return {
        'train_scores': train_scores,
        'test_scores': test_scores,
        'mean_train': np.mean(train_scores),
        'mean_test': np.mean(test_scores)
    }


def make_fp2vector(model_path: str, df: pd.DataFrame) -> np.ndarray:
    """Convert to compound vectors using FpDoc2Vec model
    
    Args:
        model_path: Path to the saved FpDoc2Vec model file
        df: DataFrame containing compound data with 'fp_3_4096' column
        
    Returns:
        NumPy array of compound vectors with shape (len(compound_vec), vector_size)
    """
    model = Doc2Vec.load(model_path)
    finger_list = list(df["fp_3_4096"])
    compound_vec = add_vectors(finger_list, model)
    vec = np.array(compound_vec)
    return vec

def main(df: pd.DataFrame, X_vec: np.ndarray, params: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Evaluate chemical categories using the provided feature vectors and classification model
    
    Args:
        df: DataFrame containing chemical compound data with category columns
        X_vec: Feature vector array for the compounds (from fingerprints, descriptors, or embeddings)
        params: Parameters dictionary for the LightGBM classifier
        
    Returns:
        Dictionary containing evaluation results for each category
    """
    
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
        y = np.array([1 if i == category else 0 for i in df[category]])
        results[category] = evaluate_category(category, X_vec, y, lightgbm_model)
    
    return results
