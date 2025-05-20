import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Tuple, Any, Optional, Union
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import f1_score
from rdkit.Chem import AllChem
from FpDoc2Vec import add_vectors, load_data, train_and_evaluate_model
from ECFP4096bit import generate_morgan_fingerprints


def FpDoc2vec(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    model_path: str, 
    lightgbm_model: lgb.LGBMClassifier,
    categories: List[str]
) -> Dict[str, Dict[str, float]]:
    """Main function to run the training and evaluation process for the FpDoc2Vec method
    
    Args:
        train_df: Training DataFrame containing molecular data with 'fp_3_4096' column
        test_df: Test DataFrame containing molecular data with 'fp_3_4096' column
        model_path: Path to the saved Doc2Vec model
        lightgbm_model: Configured LightGBM classifier instance
        categories: List of category names to evaluate
        
    Returns:
        Dictionary mapping categories to their training and test scores
    """
    # Generate fingerprint lists
    train_finger_list = list(train_df["fp_3_4096"])
    test_finger_list = list(test_df["fp_3_4096"])
    
    # Loading Doc2Vec model
    model = Doc2Vec.load(model_path)
    
    # Generate compound vectors
    train_compound_vec = add_vectors(train_finger_list, model)  
    test_compound_vec = add_vectors(test_finger_list, model)   
    X_train_vec = np.array([train_compound_vec[i] for i in range(len(train_df))])
    X_test_vec = np.array([test_compound_vec[i] for i in range(len(test_df))])
    
    # Evaluate each category
    results = {}
    for category in categories:
        results[category] = train_and_evaluate_model(
            train_df, test_df, X_train_vec, X_test_vec, category, lightgbm_model
        )
    
    return results


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
    train_desc = np.array(generate_morgan_fingerprints(train_df))
    test_desc = np.array(generate_morgan_fingerprints(test_df))
    
    # Evaluate each category
    results = {}
    for category in categories:
        results[category] = train_and_evaluate_model(
            train_df, test_df, train_desc, test_desc, category, lightgbm_model
        )
    
    return results


def descriptors(
    input_file: str, 
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    categories: List[str], 
    lightgbm_model: lgb.LGBMClassifier
) -> Dict[str, Dict[str, float]]:
    """Load chemical descriptors from file and evaluate LightGBM models for various categories
    
    Args:
        input_file: Path to the pickle file containing descriptor data
        train_df: Training DataFrame containing 'inchikey' column for matching
        test_df: Test DataFrame containing 'inchikey' column for matching
        categories: List of category names to evaluate
        lightgbm_model: Configured LightGBM classifier instance
    
    Returns:
        Dictionary mapping categories to their training and test scores
    """
    # Load dataset
    with open(input_file, "rb") as f:
        df = pickle.load(f)
        
    # Split data into train and test sets
    test_df1 = df[df["inchikey"].isin(list(test_df["inchikey"]))]
    train_df1 = df.drop(test_df1.index)
    
    # Extract descriptor columns (from column 14 onward)
    train_desc = np.array(train_df1.iloc[:, 14:])
    test_desc = np.array(test_df1.iloc[:, 14:])
    
    # Evaluate each category
    results = {}
    for category in categories:
        results[category] = train_and_evaluate_model(
            train_df1, test_df1, train_desc, test_desc, category, lightgbm_model
        )
    
    return results
  

if __name__ == "__main__":
    # Define LightGBM hyperparameters
    # Please feel free to change parameters as you like.
    params: Dict[str, Any] = {
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
    
    # Define categories to evaluate
    categories: List[str] = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin', 
        'flavouring_agent', 'agrochemical', 'volatile_oil', 'antibacterial_agent', 'insecticide'
    ]
    
    # Example paths - replace with actual paths
    input_path: str = "10genre_32descriptors.pkl"
    model_path: str = "fpdoc2vec.model"
    
    # Load data
    train_df, test_df = load_data()
    
    # Create classifier
    lightgbm_model: lgb.LGBMClassifier = lgb.LGBMClassifier(**params)
    
    # Run evaluation for different methods
    fpdoc2vec_results: Dict[str, Dict[str, float]] = FpDoc2vec(train_df, test_df, model_path, lightgbm_model, categories)
    
    ecfp_results: Dict[str, Dict[str, float]] = ECFP4096bit(train_df, test_df, categories, lightgbm_model)
    
    descriptor_results: Dict[str, Dict[str, float]] = descriptors(input_path, train_df, test_df, categories, lightgbm_model)
