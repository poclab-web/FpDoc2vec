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
