import warnings

import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier as xgb

from config import ada_params as ada_params, dt_params
from config import et_params
from config import gbm_params
from config import lr_params
from config import rf_params
from config import xgb_params
from result.classifier_performance_evaluation.evaluation import run_evaluation
from utils import save_pickle, load_pickle
from gensim.models import Doc2Vec

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Example usage - replace with your actual file paths
INPUT_PATH = "data/created_dataset/train_df.pkl"
MODEL_PATH = "model/Doc2Vec_training/fpdoc2vec.model"

result_lightgbm_path = "LightGBM_results.pkl"
result_ada_path = "AdaBoost_results.pkl"
result_et_path = "ExtraTrees_results.pkl"
result_xgb_path = "XGBoost_results.pkl"
result_rf_path = "RF_results.pkl"
result_lr_path = "LR_results.pkl"

RADIUS = 3
FP_SIZE = 4096

def main() -> None:
    """Run cross-validation evaluation for all classifiers and save the results as pickle files."""
    df = load_pickle(INPUT_PATH)
    model = Doc2Vec.load(MODEL_PATH)

    # LightGBM
    lightgbm_results = run_evaluation(df, model, RADIUS, FP_SIZE, lgb.LGBMClassifier(**gbm_params))
    save_pickle(lightgbm_results, result_lightgbm_path)

    # AdaBoost
    dt = DecisionTreeClassifier(**dt_params)
    ada_results = run_evaluation(df, model, RADIUS, FP_SIZE, AdaBoostClassifier(estimator=dt, **ada_params))
    save_pickle(ada_results, result_ada_path)

    # ExtraTrees
    et_results = run_evaluation(df, model, RADIUS, FP_SIZE, ExtraTreesClassifier(**et_params))
    save_pickle(et_results, result_et_path)

    # XGBoost
    xgb_results = run_evaluation(df, model, RADIUS, FP_SIZE, xgb(**xgb_params))
    save_pickle(xgb_results, result_xgb_path)

    # RandomForest
    rf_results = run_evaluation(df, model, RADIUS, FP_SIZE, RandomForestClassifier(**rf_params))
    save_pickle(rf_results, result_rf_path)

    # LogisticRegression
    lr_results = run_evaluation(df, model, RADIUS, FP_SIZE, LogisticRegression(**lr_params))
    save_pickle(lr_results, result_lr_path)

if __name__ == "__main__":
    main()
