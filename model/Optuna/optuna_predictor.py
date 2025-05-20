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
    model_name: str, 
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
        model_name: Model name
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
