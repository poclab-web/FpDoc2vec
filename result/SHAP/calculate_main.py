import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from gensim.models import Doc2Vec

from config.lightgbm_params import gbm_params
from utils import generate_ecfp_fingerprints, save_pickle  
from shap_core import calculate_shap_values, shap_variables



def main_ecfp(df: pd.DataFrame, purpose: str, output_path: str, max_evals: int = None) -> None:

    y = (df[purpose] == purpose).astype(int).to_numpy()
    fingerprints = generate_ecfp_fingerprints(df["ROMol"], radius=3, n_bits=4096)[0]

    model = lgb.LGBMClassifier(**gbm_params)
    model.fit(fingerprints, y)

    shap_values = calculate_shap_values(model, fingerprints, max_evals=max_evals)
    save_pickle(shap_values, output_path)


def main_fpdoc2vec(input_path: str, purpose: str, model: Doc2Vec, output_path: str, max_evals: int = 500000) -> None:
    with open(input_path, "rb") as f:
        df = pickle.load(f)

    y = np.array([1 if i == purpose else 0 for i in df[purpose]])
    fingerprints = generate_ecfp_fingerprints(df["ROMol"], radius=3, n_bits=4096)[0]

    lightgbm_model = lgb.LGBMClassifier(**gbm_params)
    pipeline, masker = shap_variables(model.dv.vectors, lightgbm_model, mask='xor')
    pipeline.fit(fingerprints, y)

    explainer = shap.Explainer(lambda x: pipeline.predict_proba(x)[:, 1], masker=masker)
    shap_values = explainer(fingerprints, max_evals=max_evals)

    save_pickle(shap_values, output_path)


if __name__ == "__main__":
    input_path="data/created_dataset/train_df.pkl"
    purpose="antioxidant"
    ecfp_output_path="antioxidant_ecfp.pkl"
    fpdoc2vec_output_path="antioxidant_fpdoc2vec.pkl"
    main_ecfp(input_path, purpose, ecfp_output_path)
    main_fpdoc2vec(input_path, purpose, fpdoc2vec_output_path, max_evals=50000)
