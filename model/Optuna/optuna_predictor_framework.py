import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
import optuna
import matplotlib.pyplot as plt
import pickle
from gensim.models.doc2vec import Doc2Vec
import optunahub
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Tuple, Callable, Any, Union, Optional, TypeVar
from optuna_doc2vec import add_vectors
import pandas as pd
from numpy.typing import NDArray
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
import xgboost as xgb

# Type variables for clearer type hints
Model = TypeVar('Model')
Params = Dict[str, Any]

def process_single_category(args: Tuple[str, NDArray, NDArray, pd.DataFrame, pd.DataFrame, Dict, Callable]) -> Tuple[float, float]:
    """
    Process a single category for parallel execution
    
    Args:
        args: Tuple containing:
            - category: Category name to process
            - X_train_vec: Training feature vectors
            - X_test_vec: Testing feature vectors
            - train_df: Training dataframe
            - test_df: Testing dataframe
            - params: Model parameters
            - model_creator: Function that creates model instance
    
    Returns:
        Tuple of (training F1 score, testing F1 score)
    """
    category, X_train_vec, X_test_vec, train_df, test_df, params, model_creator = args
    
    # Create target variables for this category
    y_train = np.array([1 if i == category else 0 for i in train_df[category]])
    y_test = np.array([1 if i == category else 0 for i in test_df[category]])
    
    # Create, train and predict with model
    model = model_creator(params)
    model.fit(X_train_vec, y_train)
    
    y_train_pred = model.predict(X_train_vec)
    y_test_pred = model.predict(X_test_vec)
    
    # Calculate scores
    train_score = f1_score(y_train, y_train_pred)
    test_score = f1_score(y_test, y_test_pred)
        
    return train_score, test_score


def create_objective_function(
    model_creator: Callable[[Dict], Model], 
    param_creator: Callable[[optuna.Trial], Dict], 
    X_train_vec: NDArray, 
    X_test_vec: NDArray, 
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    categories: List[str]
) -> Callable[[optuna.Trial], float]:
    """
    Create an objective function for Optuna
    
    Args:
        model_creator: Function that creates model instance from parameters
        param_creator: Function that creates parameter dictionary from trial
        X_train_vec: Training feature vectors
        X_test_vec: Testing feature vectors
        train_df: Training dataframe
        test_df: Testing dataframe
        categories: List of categories to predict
    
    Returns:
        Objective function for Optuna optimization
    """
    def objective(trial: optuna.Trial) -> float:
        # Generate model-specific parameter space
        params = param_creator(trial)
        
        # Prepare parallel processing
        with Pool(processes=cpu_count()) as p:
            args = [(category, X_train_vec, X_test_vec, train_df, test_df, params.copy(), model_creator)
                    for category in categories]
            results = p.map(process_single_category, args)
        
        # Aggregate scores
        train_scores, test_scores = zip(*results)
        mean_train_score = np.mean(train_scores)
        mean_test_score = np.mean(test_scores)
        
        # Save evaluation metrics
        trial.set_user_attr('train_f1', mean_train_score)
        
        return mean_test_score
    
    return objective


class ProgressCallback:
    """
    Callback to display Optuna optimization progress
    """
    def __init__(self, n_trials: int):
        """
        Args:
            n_trials: Total number of trials
        """
        self.n_trials = n_trials
        self.pbar = tqdm(total=n_trials, desc="Optimization Progress")
    
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        """
        Update progress bar after each trial
        
        Args:
            study: Optuna study object
            trial: Current trial
        """
        self.pbar.update(1)
        self.pbar.set_postfix({
            'Best Test F1': f"{study.best_value:.4f}",
            'Trial Test F1': f"{trial.value:.4f}",
            'Trial Train F1': f"{trial.user_attrs['train_f1']:.4f}"
        })


def optimize_model(
    objective: Callable[[optuna.Trial], float], 
    n_trials: int, 
    model_name: str
) -> Dict[str, Any]:
    """
    Optimize model hyperparameters
    
    Args:
        objective: Optuna objective function
        n_trials: Number of trials
        model_name: Model name (used for result filename)
    
    Returns:
        Best parameters found
    """
    # Use AutoSampler
    module = optunahub.load_module(package="samplers/auto_sampler")
    study = optuna.create_study(direction='maximize',sampler=module.AutoSampler())
    study.optimize(objective, n_trials=n_trials, callbacks=[ProgressCallback(n_trials)])
    
    print("\nBest trial:")
    print(f"  Test F1: {study.best_value:.4f}")
    print(f"  Train F1: {study.best_trial.user_attrs['train_f1']:.4f}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    return study.best_trial.params


def run_optimization(
    model_creator: Callable[[Dict], Model], 
    param_creator: Callable[[optuna.Trial], Dict], 
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    X_train_vec: NDArray, 
    X_test_vec: NDArray, 
    categories: List[str], 
    n_trials: int
) -> Dict[str, Any]:
    """
    Run the entire optimization process
    
    Args:
        model_creator: Function that creates model instance
        param_creator: Function that creates parameter space
        train_df: Training dataframe
        test_df: Testing dataframe
        X_train_vec: Training feature vectors
        X_test_vec: Testing feature vectors
        categories: List of categories to predict
        n_trials: Number of optimization trials
    
    Returns:
        Best parameters found
    """
    # Create Optuna objective function
    objective = create_objective_function(
        model_creator, param_creator, 
        X_train_vec, X_test_vec, 
        train_df, test_df, 
        categories
    )
    
    # Run parameter optimization
    return optimize_model(objective, n_trials, model_name)


def prepare_data(
    test_path: str, 
    train_path: str, 
    model_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, NDArray, NDArray, List[str]]:
    """
    Load and preprocess data
    
    Args:
        test_path: Path to test data
        train_path: Path to training data
        model_path: Path to Doc2Vec model
    
    Returns:
        Tuple containing:
            - train_df: Training dataframe
            - test_df: Testing dataframe
            - X_train_vec: Training feature vectors
            - X_test_vec: Testing feature vectors
            - categories: List of prediction categories
    """
    # Load data
    with open(test_path, "rb") as f:
        test_df = pickle.load(f)
    with open(train_path, "rb") as f:
        train_df = pickle.load(f)
    
    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye',
        'toxin', 'flavouring_agent', 'agrochemical', 'volatile_oil',
        'antibacterial_agent', 'insecticide'
    ]
    
    train_finger_list = list(train_df["fp_3_4096"])
    test_finger_list = list(test_df["fp_3_4096"])
    
    # Load Doc2Vec model
    model = Doc2Vec.load(model_path)
    
    # Generate features
    train_compound_vec = add_vectors(train_finger_list, model)
    test_compound_vec = add_vectors(test_finger_list, model)
    
    X_train_vec = np.array([train_compound_vec[i] for i in range(len(train_df))])
    X_test_vec = np.array([test_compound_vec[i] for i in range(len(test_df))])
    
    return train_df, test_df, X_train_vec, X_test_vec, categories

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

# Note: Please feel free to change the exploration range and parameters as you like.
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
    
# Note: Please feel free to change the exploration range and parameters as you like.
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

# Note: Please feel free to change the exploration range and parameters as you like.
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

# Note: Please feel free to change the exploration range and parameters as you like.
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

# Note: Please feel free to change the exploration range and parameters as you like.
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

# Note: Please feel free to change the exploration range and parameters as you like.
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

# Note: Please feel free to change the exploration range and parameters as you like.
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
    
# Note: Please feel free to change the exploration range and parameters as you like.
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

# Note: Please feel free to change the exploration range and parameters as you like.
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
