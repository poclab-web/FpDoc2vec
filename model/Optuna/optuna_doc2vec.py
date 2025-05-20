import numpy as np
import pickle
from typing import List, Tuple, Dict, Any, Union, Optional, Callable
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import optuna
import optunahub
import pandas as pd


def add_vectors(fp_list: List[List[int]], model: Doc2Vec) -> List[np.ndarray]:
    """Combine document vectors based on fingerprints
    
    Args:
        fp_list: List of fingerprint lists, where each fingerprint is represented as a list of indices
        model: Trained Doc2Vec model containing document vectors
        
    Returns:
        List of compound vectors as numpy arrays
    """
    compound_vec = []
    for i in fp_list:
        fingerprint_vec = 0
        for j in i:
            fingerprint_vec += model.dv.vectors[j]
        compound_vec.append(fingerprint_vec)
    return compound_vec


def process_single_category(args: Tuple[str, pd.DataFrame, Dict[str, Any]], purpose_description: str, dimension: int) -> Tuple[float, float]:
    """Process a single category for cross-validation
    
    Args:
        args: Tuple containing (category name, training dataframe, hyperparameters)
        purpose_description: Column name in the dataframe containing text descriptions
        dimension: Vector dimension for Doc2Vec model
        
    Returns:
        Tuple of (mean training F1 score, mean test F1 score)
    """
    category, train_df, params = args

    train_category_scores = []
    test_category_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    y = np.array([1 if i == category else 0 for i in train_df[category]])
    finger_list = list(train_df["fp_3_4096"])

    for train_idx, test_idx in skf.split(range(len(train_df)), y):
        # Prepare training data
        cm_train_df = train_df.iloc[train_idx]
        train_finger_list = list(cm_train_df["fp_3_4096"])

        # Prepare document corpus
        corpus = [sum(doc, []) for doc in cm_train_df[purpose_description]]
        tagged_documents = [TaggedDocument(words=corpus, tags=train_finger_list[i]) for i, corpus in enumerate(corpus)]

        # Train Doc2Vec model
        model = Doc2Vec(tagged_documents, vector_size=dimension, min_count=0,
                      window=params['window'],
                      min_alpha=params["min_alpha"],
                      sample=params['sample'],
                      epochs=params['epochs'],
                      negative=params['negative'],
                      ns_exponent=params['ns_exponent'],
                      workers=1, seed=0)

        # Create compound vectors
        compound_vec = add_vectors(finger_list, model)
        X_vec = StandardScaler().fit_transform(
            np.array([compound_vec[i] for i in range(len(train_df))])
        )

        # Split data
        X_train_vec, X_test_vec = X_vec[train_idx], X_vec[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train and evaluate classifier
        # Please change the classifier as you like.
        logreg = LogisticRegression(n_jobs=1)
        logreg.fit(X_train_vec, y_train)

        # Calculate scores
        y_train_pred = logreg.predict(X_train_vec)
        y_test_pred = logreg.predict(X_test_vec)

        train_category_scores.append(f1_score(y_train, y_train_pred))
        test_category_scores.append(f1_score(y_test, y_test_pred))

    return np.mean(train_category_scores), np.mean(test_category_scores)


class ProgressCallback:
    """Callback to display optimization progress
    
    Attributes:
        n_trials: Total number of trials for the optimization
        pbar: Progress bar object for visualization
    """
    def __init__(self, n_trials: int):
        self.n_trials = n_trials
        self.pbar = tqdm(total=n_trials, desc="Optimization Progress")

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Update progress bar with trial information
        
        Args:
            study: Current Optuna study object
            trial: Current Optuna trial object
        """
        self.pbar.update(1)
        self.pbar.set_postfix({
            'Best Test F1': f"{study.best_value:.4f}",
            'Trial Test F1': f"{trial.value:.4f}",
            'Trial Train F1': f"{trial.user_attrs['train_f1']:.4f}"
        })


def create_objective(train_df: pd.DataFrame, categories: List[str], purpose_description: str, dimension: int, params: Dict[str, Any]) -> Callable[[optuna.Trial], float]:
    """Create an objective function for Optuna optimization
    
    Args:
        train_df: DataFrame containing training data
        categories: List of category names to process
        purpose_description: Column name in the dataframe containing text descriptions
        dimension: Vector dimension for Doc2Vec model
        params: Dictionary containing hyperparameter search space definitions
        
    Returns:
        Objective function that takes an Optuna trial and returns the mean test F1 score
    """
    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function for hyperparameter optimization
        
        Args:
            trial: Optuna trial object for hyperparameter suggestion
            
        Returns:
            Mean test F1 score across all categories
        """
        # Define parameter search space
        trial_params = {}
        for param_name, param_spec in params.items():
            if param_spec["type"] == "int":
                trial_params[param_name] = trial.suggest_int(
                    param_name, param_spec["min"], param_spec["max"]
                )
            elif param_spec["type"] == "float":
                trial_params[param_name] = trial.suggest_float(
                    param_name, param_spec["min"], param_spec["max"]
                )
        
        # Process all categories in parallel
        with Pool(processes=cpu_count() - 1) as p:
            args = [(category, train_df, trial_params) for category in categories]
            chunk_size = max(len(categories) // (cpu_count() - 1), 1)
            scores = p.map(
                lambda x: process_single_category(x, purpose_description, dimension), 
                args, 
                chunksize=chunk_size
            )

        # Unpack scores
        train_scores = [score[0] for score in scores]
        test_scores = [score[1] for score in scores]

        # Calculate mean scores
        mean_train_score = np.mean(train_scores)
        mean_test_score = np.mean(test_scores)

        # Save training score as attribute
        trial.set_user_attr('train_f1', mean_train_score)

        return mean_test_score
    
    return objective


def optimize_doc2vec(input_traindf_path: str, categories: List[str],purpose_description: str,dimension: int,params: Dict[str, Dict[str, Any]],
                    n_trials: int) -> Dict[str, Any]:
    """Main optimization function
    
    Args:
        input_traindf_path: File path containing the training dataset
        categories: List of category names to process
        purpose_description: Column name in the dataframe containing text descriptions
        dimension: Vector dimension for Doc2Vec model
        params: Dictionary containing hyperparameter search space definitions
        n_trials: Number of optimization trials to run
        
    Returns:
        Dictionary containing the best hyperparameters found
    """
    # Load training data
    with open(input_traindf_path, "rb") as f:
        train_df = pickle.load(f)
        
    module = optunahub.load_module(package="samplers/auto_sampler")
    # Please change the sampler as you like.
    study = optuna.create_study(direction='maximize', sampler=module.AutoSampler())
    
    # Create objective function with closure over our data and parameters
    objective_func = create_objective(train_df, categories, purpose_description, dimension, params)
    
    study.optimize(objective_func, n_trials=n_trials, callbacks=[ProgressCallback(n_trials)])

    print("\nBest trial:")
    print(f"  Test F1: {study.best_value:.4f}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    return study.best_trial.params


if __name__ == "__main__":
    # Define categories to process
    categories: List[str] = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye','toxin', 'flavouring_agent', 'agrochemical', 'volatile_oil',
        'antibacterial_agent', 'insecticide'
    ]
    
    # Define hyperparameter search space
    # Please change the values as you like.
    params: Dict[str, Dict[str, Any]] = {
        'window': {'type': 'int', 'min': 3, 'max': 10},
        'min_alpha': {'type': 'float', 'min': 0.000001, 'max': 0.025},
        'sample': {'type': 'float', 'min': 0, 'max': 0.00001},
        'epochs': {'type': 'int', 'min': 30, 'max': 1000},
        'negative': {'type': 'int', 'min': 1, 'max': 20},
        'ns_exponent': {'type': 'float', 'min': 0, 'max': 1}
    }
    input_traindf_path = "train_df.pkl"
    
    # Run optimization with the loaded data
    best_params: Dict[str, Any] = optimize_doc2vec(
        input_traindf_path,
        categories=categories,
        purpose_description="description",
        dimension=100,
        params=params,
        n_trials=150
    )
