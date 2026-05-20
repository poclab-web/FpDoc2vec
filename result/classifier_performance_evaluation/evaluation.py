import pandas as pd
from typing import Any, Dict
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from utils import  fingerprints_to_vectors, generate_ecfp_fingerprints, main_cv

def run_evaluation(df: pd.DataFrame, model: Doc2Vec, radius: int, fp_size: int, classifier: Any) -> Dict[str, Dict[str, float]]:
    """Vectorize compounds with FpDoc2Vec and evaluate a classifier via cross-validation."""

    bit_list = generate_ecfp_fingerprints(df["ROMol"], radius, fp_size)[1]
    X_vec = np.array(fingerprints_to_vectors(bit_list, model))
    results = main_cv(df, X_vec, classifier)
    return results

