import pickle
import numpy as np
from typing import List, Tuple, Any, Callable, Dict, Union
from gensim.models.doc2vec import Doc2Vec
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import pandas as pd

from extratree_params import et_params
from adaboost_params import dt_params, ada_params
from lightgbm_params import gbm_params
from logistic_regression import add_vectors, train_and_evaluate_model
from randomforest_params import rf_params
from xgboost_params import xgb_params


def main(traindf_path: str, testdf_path: str, model_path: str, estimator: Any) -> None:
    """
    Main function to load data, prepare features, and evaluate models
    for different chemical categories
    
    Args:
        traindf_path: Path to pickle file containing training DataFrame
        testdf_path: Path to pickle file containing test DataFrame
        model_path: Path to saved Doc2Vec model
        estimator: ML model object with fit and predict methods
    """
    # Load data
    with open(traindf_path, "rb") as f:
        train_df = pickle.load(f)
    with open(testdf_path, "rb") as f:
        test_df = pickle.load(f)
    
    # Load model
    model = Doc2Vec.load(model_path)
    
    # Define categories
    categories = ['antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin', 
                  'flavouring_agent', 'agrochemical', 'volatile_oil', 
                  'antibacterial_agent', 'insecticide']
    
    # Prepare feature vectors
    train_finger_list = list(train_df["fp_3_4096"])
    test_finger_list = list(test_df["fp_3_4096"])
    
    train_compound_vec = addvec(train_finger_list, model)
    test_compound_vec = addvec(test_finger_list, model)
    
    X_train_vec = np.array([train_compound_vec[i] for i in range(len(train_df))])
    X_test_vec = np.array([test_compound_vec[i] for i in range(len(test_df))])
    
    # Evaluate each category
    train_scores, test_scores = [], []
    for category in categories:
        train_score, test_score = train_and_evaluate_model(
            train_df, test_df, X_train_vec, X_test_vec, category, estimator
        )
        train_scores.append(train_score)
        test_scores.append(test_score)

    print(f"Model: {type(estimator).__name__}")
    # Calculate and print average scores
    print("\n## Average scores across all categories ##")
    print(f"Average Training F1: {np.mean(train_scores):.4f}")
    print(f"Average Test F1: {np.mean(test_scores):.4f}")

if __name__ == "__main__":
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



