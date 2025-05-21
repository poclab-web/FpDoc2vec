
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from tqdm import tqdm
import pandas as pd
import lightgbm as lgb
from ECFP2048bit import add_vectors, build_doc2vec_model
from MACCSkeys import create_index_mapping, evaluate_with_keys

def process_pharmacophore_features(df: pd.DataFrame) -> Tuple[List[Optional[List[int]]], List[int]]:
    """
    Process pharmacophore features and identify invalid entries
    
    Args:
        df: DataFrame containing RDKit molecule objects in a column named 'ROMol'
        
    Returns:
        Tuple containing:
            - List of pharmacophore fingerprint bit indices (List[int]) or None for invalid entries
            - List of indices where pharmacophore generation failed
    """
    pharmacore_list = []
    for i in tqdm(df["ROMol"]):
        try:
            fp = Generate.Gen2DFingerprint(i, Gobbi_Pharm2D.factory)
            fp_bits = list(fp.GetOnBits())
            if len(fp_bits) == 0:
                pharmacore_list.append(None)
            else:
                pharmacore_list.append(fp_bits)
        except:
            print("Error")
            pharmacore_list.append(None)
            
    invalid_pharmacore_indices = []
    
    for idx, feature in enumerate(pharmacore_list):
        if feature is None:
            invalid_pharmacore_indices.append(idx)
            
    return pharmacore_list, invalid_pharmacore_indices

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
