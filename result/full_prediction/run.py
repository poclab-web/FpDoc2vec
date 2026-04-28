import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

import lightgbm as lgb
import numpy as np
import pandas as pd

# Allow importing evaluation utilities from the sibling directory
sys.path.insert(0, str(Path(__file__).parent.parent / "doc2vec_tag_evaluation"))
from evaluation import CATEGORIES, evaluate_all_categories, print_mcc_summary  

from features import make_descriptor, make_ecfp, make_fp2vector, make_name2vector

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ---------------------------------------------------------------------------
# Settings — edit paths and parameters here
# ---------------------------------------------------------------------------

DATA_PATH: str = "../../data/created_dataset/train_df.pkl"
FPDOC2VEC_MODEL_PATH: str = "fpdoc2vec.model"
NAMEDOC2VEC_MODEL_PATH: str = "namedoc2vec.model"
DESCRIPTOR_PATH: str = "../../data/created_dataset/10genre_32descriptor.pkl"

LIGHTGBM_PARAMS: Dict[str, Any] = {
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
    "objective": "binary",
    "random_state": 50,
    "verbose": -1,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_results(results: Dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(results, f)


def run_method(name: str, X: np.ndarray, df: pd.DataFrame, lgbm: lgb.LGBMClassifier) -> Dict:
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    results = evaluate_all_categories(X, df, lgbm)
    print_mcc_summary(results, name)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    lgbm = lgb.LGBMClassifier(**LIGHTGBM_PARAMS)

    all_results = {}

    # 1. FpDoc2Vec
    X_fp = make_fp2vector(FPDOC2VEC_MODEL_PATH, df)
    all_results["FpDoc2Vec"] = run_method("FpDoc2Vec", X_fp, df, lgbm)
    save_results(all_results["FpDoc2Vec"], "results/fpdoc2vec.pkl")

    # 2. NameDoc2Vec
    X_name = make_name2vector(NAMEDOC2VEC_MODEL_PATH, df)
    all_results["NameDoc2Vec"] = run_method("NameDoc2Vec", X_name, df, lgbm)
    save_results(all_results["NameDoc2Vec"], "results/namedoc2vec.pkl")

    # 3. ECFP (radius=3, 4096 bits)
    X_ecfp = make_ecfp(df, radius=3, n_bits=4096)
    all_results["ECFP"] = run_method("ECFP (radius=3, 4096 bits)", X_ecfp, df, lgbm)
    save_results(all_results["ECFP"], "results/ecfp.pkl")

    # 4. Descriptors
    X_desc = make_descriptor(DESCRIPTOR_PATH)
    all_results["Descriptors"] = run_method("Descriptors", X_desc, df, lgbm)
    save_results(all_results["Descriptors"], "results/descriptors.pkl")

    # Overall summary
    print("\n" + "=" * 50)
    print("  OVERALL SUMMARY (mean test MCC)")
    print("=" * 50)
    for method, results in all_results.items():
        mcc_scores = [results[cat]["mcc"]["mean_test"] for cat in CATEGORIES if cat in results]
        print(f"  {method:20s}: {np.mean(mcc_scores):.4f}")
