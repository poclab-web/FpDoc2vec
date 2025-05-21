import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import List, Dict

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
