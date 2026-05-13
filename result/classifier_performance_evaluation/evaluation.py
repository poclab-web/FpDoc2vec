import pickle
from typing import Dict, List, Union

import numpy as np
from gensim.models.doc2vec import Doc2Vec

from util import CATEGORIES, fingerprints_to_vectors, evaluate_category_cv, generate_ecfp_fingerprints, main_cv

# def run_evaluation(input_path: str, model_path: str, radius: int, fp_size: int, classifier) -> Dict[str, Dict[str, float]]:
#     with open(input_path, "rb") as f:
#         df = pickle.load(f)
#     model = Doc2Vec.load(model_path)

#     bit_list = generate_ecfp_fingerprints(df, radius, fp_size)[1]
#     X_vec = np.array(fingerprints_to_vectors(bit_list, model))

#     results = {}
#     for category in CATEGORIES:
#         y = (df[category] == category).astype(int).to_numpy()
#         results[category] = evaluate_category_cv(X_vec, y, classifier)
#     return results

def run_evaluation(input_path: str, model_path: str, radius: int, fp_size: int, classifier) -> Dict[str, Dict[str, float]]:
    with open(input_path, "rb") as f:
        df = pickle.load(f)
    model = Doc2Vec.load(model_path)

    bit_list = generate_ecfp_fingerprints(df, radius, fp_size)[1]
    X_vec = np.array(fingerprints_to_vectors(bit_list, model))

    results = main_cv(df, X_vec, classifier)
    return results

