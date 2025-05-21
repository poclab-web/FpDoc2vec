import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from xgboost import XGBClassifier as xgb

from extratree_params import et_params
from adaboost_params import dt_params, ada_params
from lightgbm_params import gbm_params
from logistic_regression import add_vectors, train_and_evaluate_model, main
from randomforest_params import rf_params
from xgboost_params import xgb_params

# Example usage - replace with your actual file paths
traindf_path = "train_df.pkl"
testdf_path = "test_df.pkl"
# Model for a novel compound
model_path = "fpdoc2vec_novel.model"

#LR
logreg = LogisticRegression(random_state = 50)
main(traindf_path, testdf_path, model_path, logreg)
#LightGBM
lightgbm = lgb.LGBMClassifier(**gbm_params)
main(traindf_path, testdf_path, model_path, lightgbm)
#ExtraTree
et = ExtraTreesClassifier(**et_params)
main(traindf_path, testdf_path, model_path, et)
#RandomForest
rf = RandomForestClassifier(**rf_params)
main(traindf_path, testdf_path, model_path, rf)
#XGboost
xg_boost = xgb(**xgb_params)
main(traindf_path, testdf_path, model_path, xg_boost)
#adaboost
dt = DecisionTreeClassifier(**dt_params)
ada = AdaBoostClassifier(estimator=dt, **ada_params)
main(traindf_path, testdf_path, model_path, ada)



