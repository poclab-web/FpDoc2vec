import pickle
import os
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from ECFP2048bit import generate_morgan_fingerprints, add_vectors, evaluate_category, build_doc2vec_model, main
from MACCSkeys import generate_maccs_fingerprints, create_index_mapping, evaluate_with_keys, main_maccs
from pharmacophore import process_pharmacophore_features, main_pharma
from smiles_to_ngram import smiles_to_ngrams, make_ngramlist, make_ngramlist


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
