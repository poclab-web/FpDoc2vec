import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from gensim.models.doc2vec import Doc2Vec
from Descriptors import make_descriptor
from ECFP4096bit import generate_morgan_fingerprints
from FpDoc2vec import add_vectors, evaluate_category, make_fp2vector, main
from NameDoc2Vec import make_name2vector


# Model hyperparameters
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

# Example paths - replace with actual paths
input_path: str = "10genre_dataset.pkl"

# Load dataset
with open(input_path, "rb") as f:
    df: pd.DataFrame = pickle.load(f)

# FP Doc2Vec approach
# Example paths - replace with actual paths
fp_model_path: str = "fpdoc2vec.model"
fpvec: np.ndarray = make_fp2vector(model_path=fp_model_path, df=df)

# Loading Doc2Vec model
fp_model: Doc2Vec = Doc2Vec.load(fp_model_path)
fp_results: Dict[str, Dict[str, float]] = main(df=df, X_vec=fpvec, params=params)

# Name Doc2Vec approach
# Example paths - replace with actual paths
name_model_path: str = "namedoc2vec.model"
namevec: np.ndarray = make_name2vector(model_path=name_model_path, df=df)

# Loading Doc2Vec model
name_model: Doc2Vec = Doc2Vec.load(name_model_path)
name_results: Dict[str, Dict[str, float]] = main(df=df, X_vec=namevec, params=params)

# ECFP approach
ecfp: np.ndarray = np.array(generate_morgan_fingerprints(df, 3, 4096))
ecfp_results: Dict[str, Dict[str, float]] = main(df=df, X_vec=ecfp, params=params)

# Descriptor approach
# Example paths - replace with actual paths
input_descriptor_path: str = "10genre_32descriptor.pkl"
desc: np.ndarray = make_descriptor(input_path=input_descriptor_path)
desc_results: Dict[str, Dict[str, float]] = main(df=df, X_vec=desc, params=params)
