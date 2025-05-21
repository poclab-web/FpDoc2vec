import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
import xgboost as xgb
import optuna
from sklearn.metrics import f1_score
from tqdm import tqdm
import pickle
from gensim.models.doc2vec import Doc2Vec
import optunahub
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Tuple, Callable, Any, Union, Optional, TypeVar
from optuna_doc2vec import add_vectors
import pandas as pd
from numpy.typing import NDArray
from optuna_predictor_framework import process_single_category, create_objective_function, ProgressCallback, optimize_model, run_optimization, prepare_data, create_lightgbm_model, create_lightgbm_params, create_adaboost_model, create_adaboost_params,  create_rf_model, create_rf_params, create_xgboost_model, create_xgboost_params, create_et_model, create_et_params,  

# Note: Please feel free to change the exploration range and parameters as you like.

def create_lightgbm_model(params: Dict[str, Any]) -> lgb.LGBMClassifier:
    """
    Create a LightGBM model with given parameters
    
    Args:
        params: Dictionary of model parameters
    
    Returns:
        Initialized LightGBM model
    """
    # Pass all parameters directly to the classifier
    return lgb.LGBMClassifier(objective='binary',random_state=0,**params)


def create_lightgbm_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Create parameter space for LightGBM optimization
    
    Args:
        trial: Optuna trial object
    
    Returns:
        Dictionary of parameters to optimize
    """
    # First determine the boosting type
    boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])

    # Set basic parameters
    params = {
        'boosting_type': boosting_type,
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'num_leaves': trial.suggest_int('num_leaves', 10, 50),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0),
    }

    # Add parameters based on boosting type
    if boosting_type == 'goss':
        params.update({
            'top_rate': trial.suggest_float('top_rate', 0.1, 0.3),
            'other_rate': trial.suggest_float('other_rate', 0.1, 0.2)
        })
    else:
        # Add bagging parameters for non-GOSS boosting types
        params.update({
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 7)
        })

    # Add dropout parameters for dart boosting
    if boosting_type == 'dart':
        params.update({
            'drop_rate': trial.suggest_float('drop_rate', 0.1, 0.5),
            'skip_drop': trial.suggest_float('skip_drop', 0.1, 0.5)
        })
    
    return params

def create_adaboost_model(params: Dict[str, Any]) -> AdaBoostClassifier:
    """
    Create an AdaBoost model with given parameters
    
    Args:
        params: Dictionary of model parameters
    
    Returns:
        Initialized AdaBoost model
    """
    # Extract base estimator parameters
    base_max_depth = params.pop('base_max_depth', 1)
    base_class_weight = params.pop('base_class_weight', None)
    
    # Create base estimator
    base_estimator = DecisionTreeClassifier(
        max_depth=base_max_depth,
        class_weight=base_class_weight
    )
    
    # Create and return AdaBoost model
    return AdaBoostClassifier(estimator=base_estimator,random_state=0, **params)


def create_adaboost_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Create parameter space for AdaBoost optimization
    
    Args:
        trial: Optuna trial object
    
    Returns:
        Dictionary of parameters to optimize
    """
    params = {
        # AdaBoost specific parameters
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'algorithm': trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R']),
        
        # Base estimator (DecisionTree) parameters
        'base_max_depth': trial.suggest_int('base_max_depth', 1, 10),
        'base_class_weight': trial.suggest_categorical('base_class_weight', ['balanced', None]),
    }
    
    return params

def create_rf_model(params: Dict[str, Any]) -> RandomForestClassifier:
    """
    Create a RandomForest model with given parameters
    
    Args:
        params: Dictionary of model parameters
    
    Returns:
        Initialized RandomForest model
    """
    # Handle max_features special case
    max_features = params.pop('max_features')
    if max_features is None and 'max_features_ratio' in params:
        max_features = params.pop('max_features_ratio')
    
    # Create and return model
    return RandomForestClassifier(max_features=max_features,random_state=0,n_jobs=-1,verbose=0,**params)


def create_rf_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Create parameter space for RandomForest optimization
    
    Args:
        trial: Optuna trial object
    
    Returns:
        Dictionary of parameters to optimize
    """
    params = {
        # Number of trees - wide range
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        
        # Split criterion
        'criterion': trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        
        # Tree depth parameters
        'max_depth': trial.suggest_int('max_depth', 3, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
        
        # Feature selection
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        
        # Bootstrap parameters
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        
        # Class weight
        'class_weight': trial.suggest_categorical('class_weight',
                                                ['balanced', 'balanced_subsample', None]),
    }
    
    # Add max_samples only if bootstrap is True
    if params['bootstrap']:
        params['max_samples'] = trial.suggest_float('max_samples', 0.5, 1.0)
    else:
        params['max_samples'] = None
    
    # Handle max_features=None case
    if params['max_features'] is None:
        params['max_features_ratio'] = trial.suggest_float('max_features_ratio', 0.1, 1.0)
    
    return params

def create_xgboost_model(params: Dict[str, Any]) -> xgb.XGBClassifier:
    """
    Create an XGBoost model with given parameters
    
    Args:
        params: Dictionary of model parameters
    
    Returns:
        Initialized XGBoost model
    """
    # Handle special parameters if needed
    booster = params.pop('booster', 'gbtree')
    
    # Create and return model
    return xgb.XGBClassifier(booster=booster,random_state=0, **params)


def create_xgboost_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Create parameter space for XGBoost optimization
    
    Args:
        trial: Optuna trial object
    
    Returns:
        Dictionary of parameters to optimize
    """
    params = {
        # Booster parameters
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        
        # Tree parameters
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        
        # Regularization parameters
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        
        # Other parameters
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10.0),
    }
    
    # Add booster-specific parameters
    if params['booster'] == 'dart':
        params.update({
            'sample_type': trial.suggest_categorical('sample_type', ['uniform', 'weighted']),
            'normalize_type': trial.suggest_categorical('normalize_type', ['tree', 'forest']),
            'rate_drop': trial.suggest_float('rate_drop', 0.0, 0.5),
            'skip_drop': trial.suggest_float('skip_drop', 0.0, 0.5),
        })
    
    return params

def create_et_model(params: Dict[str, Any]) -> ExtraTreesClassifier:
    """
    Create an ExtraTrees model with given parameters
    
    Args:
        params: Dictionary of model parameters
    
    Returns:
        Initialized ExtraTrees model
    """
    # Handle max_features parameter special case
    max_features = params.pop('max_features', None)
    if max_features is None and 'max_features_ratio' in params:
        max_features = params.pop('max_features_ratio')
    
    # Handle bootstrap parameter and related max_samples
    bootstrap = params.pop('bootstrap', True)
    max_samples = params.pop('max_samples', None) if bootstrap else None
    
    # Create and return model
    return ExtraTreesClassifier(
        max_features=max_features,
        bootstrap=bootstrap,
        max_samples=max_samples,
        random_state=0,
        n_jobs=-1,
        verbose=0,
        **params
    )


def create_et_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Create parameter space for ExtraTrees optimization
    
    Args:
        trial: Optuna trial object
    
    Returns:
        Dictionary of parameters to optimize
    """
    params = {
        # Number of trees
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        
        # Split criterion
        'criterion': trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        
        # Tree depth parameters
        'max_depth': trial.suggest_int('max_depth', 3, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
        
        # Feature selection
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        
        # Bootstrap options
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        
        # Class balance
        'class_weight': trial.suggest_categorical('class_weight',
                                               ['balanced', 'balanced_subsample', None]),
    }
    
    # Add bootstrap-dependent parameters
    if params['bootstrap']:
        params['max_samples'] = trial.suggest_float('max_samples', 0.5, 1.0)
    
    # Add max_features-dependent parameters
    if params['max_features'] is None:
        params['max_features_ratio'] = trial.suggest_float('max_features_ratio', 0.1, 1.0)
    
    return params

if __name__ == "__main__":
    # Prepare data
    # Example usage - replace with your actual file paths
    test_data_path = "test_df.pkl"
    train_data_path = "train_df.pkl"
    model_path = "fpdoc2vec.model"
    train_df, test_df, X_train_vec, X_test_vec, categories = prepare_data(
        test_path=test_data_path,
        train_path=train_data_path,
        model_path=model_path
    )
    
    # Run optimization
    # The following is that provides examples of LightGBM.
    best_params = run_optimization(
        model_creator=create_lightgbm_model, # Please appropriately change this part.
        param_creator=create_lightgbm_params, # Please appropriately change this part.
        train_df=train_df,
        test_df=test_df,
        X_train_vec=X_train_vec,
        X_test_vec=X_test_vec,
        categories=categories,
        n_trials=300  # You can adjust the number of trials
    )
