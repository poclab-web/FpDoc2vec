import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from rdkit.Chem import AllChem
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from ECFP2048bit import generate_morgan_fingerprints, add_vectors, evaluate_category, build_doc2vec_model
from MACCSkeys import generate_maccs_fingerprints, create_index_mapping, evaluate_with_keys
from pharmacophore import process_pharmacophore_features
from smiles_to_ngram import smiles_to_ngrams, make_ngramlist


def main(input_path: str, feature_list: List[Any], doc2vec_param: Dict[str, Any], 
         lightgbm_model: lgb.LGBMClassifier, purpose_description: str = "description_remove_stop_words") -> Dict[str, Dict[str, float]]:
    """
    Main function to train and evaluate compound classification models using provided features and Doc2Vec.
    
    Args:
        input_path: Path to the pickle file containing compound data
        feature_list: List of molecular features (like fingerprints) to use in the model
        doc2vec_param: Parameters for the Doc2Vec model
        lightgbm_model: Pre-configured LightGBM classifier
        purpose_description: Column name in the DataFrame containing text descriptions
        
    Returns:
        Dictionary mapping category names to evaluation results
    """
    # Load dataset
    with open(input_path, "rb") as f:
        df = pickle.load(f)
        
    # Define categories to evaluate
    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin', 
        'flavouring_agent', 'agrochemical', 'volatile_oil', 'antibacterial_agent', 'insecticide'
    ]
    
    # Prepare corpus for Doc2Vec
    corpus = [sum(doc, []) for doc in df[purpose_description]]
    
    # Build Doc2Vec model
    model = build_doc2vec_model(corpus, feature_list, doc2vec_param)
    
    # Generate compound vectors
    compound_vec = add_vectors(feature_list, model)
    X_vec = np.array([compound_vec[i] for i in range(len(df))])
    
    
    # Evaluate each category
    results = {}
    for category in categories:
        y = np.array([1 if i == category else 0 for i in df[category]])
        results[category] = evaluate_category(category, X_vec, y, lightgbm_model)

    return results


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


def main_pharma(input_path: str, params: Dict[str, Any], lightgbm_model: lgb.LGBMClassifier,
                purpose_description: str = "description_remove_stop_words") -> Dict[str, Dict[str, float]]:
    """
    Train and evaluate models using pharmacophore features.
    
    Args:
        input_path: Path to the pickle file containing compound data
        params: Parameters for the Doc2Vec model
        lightgbm_model: Pre-configured LightGBM classifier
        purpose_description: Column name in the DataFrame containing text descriptions
        
    Returns:
        Dictionary mapping category names to evaluation results
    """
    # Load dataset
    with open(input_path, "rb") as f:
        df = pickle.load(f)
    
    # Process pharmacophore features
    pharmacore_list, invalid_pharmacore_indices = process_pharmacophore_features(df)    
  
    df["pharmacore"] = pharmacore_list
    
    # Create a filtered dataframe with valid pharmacophore features
    df_pharm = df.copy()
    df_pharm = df_pharm.drop(invalid_pharmacore_indices).reset_index(drop=True)
    print(f"Number of compounds with valid pharmacophore features: {len(df_pharm)}")
    
    # Prepare data for Doc2Vec
    pharm_list = list(df_pharm["pharmacore"])
    corpus = [sum(doc, []) for doc in df_pharm[purpose_description]]
    
    # Define categories
    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin',
        'flavouring_agent', 'agrochemical', 'volatile_oil', 'antibacterial_agent', 'insecticide'
    ]
    
    # Build Doc2Vec model
    model = build_doc2vec_model(corpus, pharm_list, params)
    
    # Generate compound vectors
    compound_vec = add_vectors(pharm_list, model)
    X_vec_pharm = np.array([compound_vec[i] for i in range(len(df_pharm))])
    
    # Create index mapping
    index_mapping = create_index_mapping(len(df), invalid_pharmacore_indices)
    
    # Evaluate model performance
    results = evaluate_with_keys(lightgbm_model, df, df_pharm, X_vec_pharm, categories, index_mapping)
    return results
    

if __name__ == "__main__":
    # Example usage - replace with your actual params
    doc2vec_param: Dict[str, Any] = {
        "vector_size": 100, 
        "min_count": 0,
        "window": 10,
        "min_alpha": 0.023491749982816976,
        "sample": 7.343338709169564e-06,
        "epochs": 859,
        "negative": 2,
        "ns_exponent": 0.8998927133390002,
        "workers": 1, 
        "seed": 100
    }
  
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

    # Create classifier
    lightgbm_model = lgb.LGBMClassifier(**gbm_params)
    
    # Load dataset for later use
    # Example usage - replace with your actual file paths
    input_path = "10genre_dataset.pkl"
    with open(input_path, "rb") as f:
        df = pickle.load(f)

    # Tag 2048ECFP
    df["fp_2_2048"] = generate_morgan_fingerprints(df, 2, 2048)
    finger_list = list(df["fp_2_2048"])
    results = main(input_path, finger_list, doc2vec_param, lightgbm_model)

    # Tag 4096ECFP
    df["fp_3_4096"] = generate_morgan_fingerprints(df, 3, 4096)
    finger_list = list(df["fp_3_4096"])
    results = main(input_path, finger_list, doc2vec_param, lightgbm_model)

    # Tag Maccs keys
    results = main_maccs(input_path, doc2vec_param, lightgbm_model)

    # Tag Pharmacophore features
    results = main_pharma(input_path, doc2vec_param, lightgbm_model)

    # Tag smiles_ngram
    ngram_list = make_ngramlist(input_path, n=3)
    results = main(input_path, ngram_list, doc2vec_param, lightgbm_model)
