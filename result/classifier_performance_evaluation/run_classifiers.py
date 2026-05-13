import pickle
import warnings

import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier as xgb

from config.adaboost_params import ada_params, dt_params
from config.extratree_params import et_params
from config.lightgbm_params import gbm_params
from config.logistic_regression_params import lr_params
from config.randomforest_params import rf_params
from config.xgboost_params import xgb_params
from result.classifier_performance_evaluation.evaluation import run_evaluation

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Example usage - replace with your actual file paths
INPUT_PATH = "train_df.pkl"
MODEL_PATH = "fpdoc2vec.model"
RADIUS = 3
FP_SIZE = 4096


def save(results, name: str) -> None:
    with open(f"{name}_results.pkl", "wb") as f:
        pickle.dump(results, f)


def main():
    # LightGBM
    lightgbm_results = run_evaluation(INPUT_PATH, MODEL_PATH, RADIUS, FP_SIZE, lgb.LGBMClassifier(**gbm_params))
    save(lightgbm_results, "LightGBM")

    # AdaBoost
    dt = DecisionTreeClassifier(**dt_params)
    ada_results = run_evaluation(INPUT_PATH, MODEL_PATH, RADIUS, FP_SIZE, AdaBoostClassifier(estimator=dt, **ada_params))
    save(ada_results, "AdaBoost")

    # ExtraTrees
    et_results = run_evaluation(INPUT_PATH, MODEL_PATH, RADIUS, FP_SIZE, ExtraTreesClassifier(**et_params))
    save(et_results, "ExtraTrees")

    # XGBoost
    xgb_results = run_evaluation(INPUT_PATH, MODEL_PATH, RADIUS, FP_SIZE, xgb(**xgb_params))
    save(xgb_results, "XGBoost")

    # RandomForest
    rf_results = run_evaluation(INPUT_PATH, MODEL_PATH, RADIUS, FP_SIZE, RandomForestClassifier(**rf_params))
    save(rf_results, "RF")

    # LogisticRegression
    lr_results = run_evaluation(INPUT_PATH, MODEL_PATH, RADIUS, FP_SIZE, LogisticRegression(**lr_params))
    save(lr_results, "LR")

if __name__ == "__main__":
    main()
