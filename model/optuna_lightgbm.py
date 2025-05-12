import numpy as np
import pickle
import lightgbm as lgb
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import f1_score
from multiprocessing import Pool, cpu_count
import optuna
import optunahub
from tqdm import tqdm

def addvec(fingerprint_df, model):
    """
    Create compound vectors by summing Doc2Vec vectors for each fingerprint
    """
    compound_vec = []
    for i in fingerprint_df:
        fingerprint_vec = 0
        for j in i:
            fingerprint_vec += model.dv.vectors[j]
        compound_vec.append(fingerprint_vec)
    return compound_vec

def process_single_category(args):
    """
    Train and evaluate LightGBM model for a single category
    """
    category, X_train_vec, X_test_vec, train_df, test_df, params = args

    # Create target variables for each category
    y_train = np.array([1 if i == category else 0 for i in train_df[category]])
    y_test = np.array([1 if i == category else 0 for i in test_df[category]])

    # Train and predict with model
    lightgbm = lgb.LGBMClassifier(**params)
    lightgbm.fit(X_train_vec, y_train)

    y_train_pred = lightgbm.predict(X_train_vec)
    y_test_pred = lightgbm.predict(X_test_vec)

    # Calculate scores
    train_score = f1_score(y_train, y_train_pred)
    test_score = f1_score(y_test, y_test_pred)

    return train_score, test_score

def objective(trial):
    """
    Objective function for Optuna optimization
    """
    # Determine boosting type
    boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])

    # Set basic parameters
    params = {
        'boosting_type': boosting_type,
        'num_leaves': trial.suggest_int('num_leaves', 10, 35),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        "objective": 'binary',
        'class_weight': 'balanced',
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.5),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1e-1, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
        "random_state": 0
    }
    
    # Additional parameters based on boosting type
    if params['boosting_type'] != 'goss':
        params['subsample_freq'] = trial.suggest_int('subsample_freq', 0, 5)

    if boosting_type == 'goss':
        params.update({
            'top_rate': trial.suggest_float('top_rate', 0.1, 0.3),
            'other_rate': trial.suggest_float('other_rate', 0.05, 0.2)
        })

    if boosting_type == 'dart':
        params.update({
            'drop_rate': trial.suggest_float('drop_rate', 0.1, 0.3),
            'skip_drop': trial.suggest_float('skip_drop', 0.3, 0.7)
        })

    # Parallel processing
    with Pool(processes=cpu_count()) as p:
        args = [(category, X_train_vec, X_test_vec, train_df, test_df, params)
                for category in categories]
        results = p.map(process_single_category, args)

    # Aggregate scores
    train_scores, test_scores = zip(*results)
    mean_train_score = np.mean(train_scores)
    mean_test_score = np.mean(test_scores)

    # Save evaluation metrics
    trial.set_user_attr('train_f1', mean_train_score)

    return mean_test_score

class ProgressCallback:
    """
    Callback to display optimization progress
    """
    def __init__(self, n_trials):
        self.n_trials = n_trials
        self.pbar = tqdm(total=n_trials, desc="Optimization Progress")

    def __call__(self, study, trial):
        self.pbar.update(1)
        self.pbar.set_postfix({
            'Best Test F1': f"{study.best_value:.4f}",
            'Trial Test F1': f"{trial.value:.4f}",
            'Trial Train F1': f"{trial.user_attrs['train_f1']:.4f}"
        })

def optimize_lightgbm(n_trials):
    """
    Run hyperparameter optimization for LightGBM models
    """
    module = optunahub.load_module(package="samplers/auto_sampler")
    study = optuna.create_study(direction='maximize', sampler=module.AutoSampler())
    study.optimize(objective, n_trials=n_trials, callbacks=[ProgressCallback(n_trials)])

    print("\nBest trial:")
    print(f"  Test F1: {study.best_value:.4f}")
    print(f"  Train F1: {study.best_trial.user_attrs['train_f1']:.4f}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    return study.best_trial.params


if __name__ == "__main__":
    # Load and preprocess data
    with open("data/test_df.pkl", "rb") as f:
        test_df = pickle.load(f)
    with open("data/train_df.pkl", "rb") as f:
        train_df = pickle.load(f)

    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye',
        'toxin', 'flavouring_agent', 'agrochemical', 'volatile_oil',
        'antibacterial_agent', 'insecticide'
    ]

    train_finger_list = list(train_df["fp_3_4096"])
    test_finger_list = list(test_df["fp_3_4096"])

    model = Doc2Vec.load("model/20250303_fp4096.model")

    # Generate features and scale
    train_compound_vec = addvec(train_finger_list, model)
    test_compound_vec = addvec(test_finger_list, model)

    X_train_vec = np.array([train_compound_vec[i] for i in range(len(train_df))])
    X_test_vec = np.array([test_compound_vec[i] for i in range(len(test_df))])

    # Run parameter optimization
    best_params = optimize_lightgbm(500)
