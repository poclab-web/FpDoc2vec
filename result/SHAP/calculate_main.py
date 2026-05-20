import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from gensim.models import Doc2Vec

from config.lightgbm_params import gbm_params
from utils import generate_ecfp_fingerprints, save_pickle, load_pickle
from shap_core import calculate_shap_values, shap_variables


def main_ecfp(fingerprints: np.ndarray, y: np.ndarray, output_path: str, max_evals: int = None) -> None:
    """Train a LightGBM model on ECFP fingerprints and save the resulting SHAP values."""
    model = lgb.LGBMClassifier(**gbm_params)
    model.fit(fingerprints, y)

    shap_values = calculate_shap_values(model, fingerprints, max_evals=max_evals)
    save_pickle(shap_values, output_path)


def main_fpdoc2vec(fingerprints: np.ndarray, y: np.ndarray, model: Doc2Vec, output_path: str, max_evals: int = 500000) -> None:
    """Train a LightGBM pipeline with FpDoc2Vec embeddings and save the resulting SHAP values."""
    lightgbm_model = lgb.LGBMClassifier(**gbm_params)
    pipeline, masker = shap_variables(model.dv.vectors, lightgbm_model, mask='xor')
    pipeline.fit(fingerprints, y)

    explainer = shap.Explainer(lambda x: pipeline.predict_proba(x)[:, 1], masker=masker)
    shap_values = explainer(fingerprints, max_evals=max_evals)

    save_pickle(shap_values, output_path)


if __name__ == "__main__":
    purpose = "antioxidant"
    data_path = "data/created_dataset/train_df.pkl"
    ecfp_shap_result_path = "antioxidant_ecfp.pkl"
    fpdoc2vec_shap_result_path = "antioxidant_fpdoc2vec.pkl"
    fpdoc2ec_path = "model/Doc2Vec_training/fpdoc2vec.model"

    df = load_pickle(data_path)
    y = (df[purpose] == purpose).astype(int).to_numpy()
    fingerprints = generate_ecfp_fingerprints(list(df["ROMol"]), radius=3, n_bits=4096)[0]
    # calculate SHAP values for ECFP
    main_ecfp(fingerprints, y, ecfp_shap_result_path)
    # calculate SHAP values for FpDoc2Vec
    doc2vec_model = Doc2Vec.load(fpdoc2ec_path)
    main_fpdoc2vec(fingerprints, y, doc2vec_model, fpdoc2vec_shap_result_path, max_evals=50000)
