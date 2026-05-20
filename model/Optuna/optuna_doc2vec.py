import warnings
from multiprocessing import Pool, cpu_count
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import optuna
import optunahub
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from optuna import Study, Trial
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from utils import (
    CATEGORIES,
    generate_ecfp_fingerprints,
    build_tagged_corpus,
    fingerprints_to_vectors,
    load_pickle,
    save_pickle,
)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def process_single_fold(args: tuple[str, pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray, int]) -> tuple[str, int, float, float]:
    """Train a Doc2Vec model and evaluate a logistic regression classifier for one category and one cross-validation fold, returning train/test F1 scores."""
    category, train_df, params, train_idx, test_idx, fold_idx = args

    y = (train_df[category] == category).astype(int).to_numpy()

    finger_list = generate_ecfp_fingerprints(list(train_df["ROMol"]), radius=3, n_bits=4096)[1]

    fold_train_df = train_df.iloc[train_idx]
    fold_finger_list = generate_ecfp_fingerprints(list(fold_train_df["ROMol"]), radius=3, n_bits=4096)[1]

    corpus = build_tagged_corpus(fold_train_df, fold_finger_list, "description_gensim")

    model = Doc2Vec(
        corpus,
        dm=params["dm"],
        vector_size=params["vector_size"],
        min_count=params["min_count"],
        window=params["window"],
        alpha=params["alpha"],
        sample=params["sample"],
        epochs=params["epochs"],
        negative=params["negative"],
        workers=1,
        seed=0,
    )

    X = fingerprints_to_vectors(finger_list, model)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)

    train_f1 = f1_score(y_train, clf.predict(X_train))
    test_f1 = f1_score(y_test, clf.predict(X_test))

    return category, fold_idx, train_f1, test_f1


def make_objective(train_df: pd.DataFrame, categories: list[str]) -> Callable[[Trial], float]:
    """Build and return an Optuna objective function that performs stratified k-fold cross-validation over all categories using multiprocessing."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    def objective(trial: Trial) -> float:
        vector_size = trial.suggest_int("vector_size", 50, 150, step=10)
        dm = trial.suggest_categorical("dm", [0, 1])
        if dm == 0:  # DBOW
            window = trial.suggest_int("window", 6, 20)
            epochs = trial.suggest_int("epochs", 50, 150, step=5)
        else:  # DM
            window = trial.suggest_int("window", 3, 10)
            epochs = trial.suggest_int("epochs", 100, 1000, step=10)

        params = {
            "vector_size": vector_size,
            "dm": dm,
            "window": window,
            "min_count": 0,
            "alpha": trial.suggest_float("alpha", 0.01, 0.05, log=True),
            "sample": trial.suggest_float("sample", 1e-6, 1e-4, log=True),
            "epochs": epochs,
            "negative": trial.suggest_int("negative", 5, 20),
        }

        all_tasks = []
        for category in categories:
            y = np.array([1 if label == category else 0 for label in train_df[category]])
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(range(len(train_df)), y)):
                all_tasks.append((category, train_df, params, train_idx, test_idx, fold_idx))

        with Pool(processes=cpu_count()) as pool:
            results = pool.map(process_single_fold, all_tasks)

        category_results = {cat: {"train": [], "test": []} for cat in categories}
        for category, _, train_score, test_score in results:
            category_results[category]["train"].append(train_score)
            category_results[category]["test"].append(test_score)

        train_mean = np.mean([np.mean(category_results[cat]["train"]) for cat in categories])
        test_mean = np.mean([np.mean(category_results[cat]["test"]) for cat in categories])

        trial.set_user_attr("train_f1", train_mean)
        return test_mean

    return objective


class ProgressCallback:
    def __init__(self, n_trials: int) -> None:
        """Initialize a tqdm progress bar for tracking Optuna optimization trials."""
        self.pbar = tqdm(total=n_trials, desc="Optimization Progress")

    def __call__(self, study: Study, trial: Trial) -> None:
        """Update the progress bar with the current trial's train/test F1 scores."""
        self.pbar.update(1)
        self.pbar.set_postfix({
            "Best Test F1": f"{study.best_value:.4f}",
            "Trial Test F1": f"{trial.value:.4f}",
            "Trial Training F1": f"{trial.user_attrs['train_f1']:.4f}",
        })

    def close(self) -> None:
        """Close the progress bar."""
        self.pbar.close()


def optimize_doc2vec(train_df: pd.DataFrame, categories: list[str], n_trials: int, output_path: str) -> dict[str, Any]:
    """Run Optuna hyperparameter optimization for Doc2Vec, plot the F1 score history, and return the best parameters."""
    module = optunahub.load_module(package="samplers/auto_sampler")
    study = optuna.create_study(direction="maximize", sampler=module.AutoSampler())

    callback = ProgressCallback(n_trials)
    study.optimize(make_objective(train_df, categories), n_trials=n_trials, callbacks=[callback])
    callback.close()

    best = study.best_trial
    print("\nBest trial:")
    print(f"  Test F1:  {study.best_value:.4f}")
    print(f"  Train F1: {best.user_attrs['train_f1']:.4f}")
    print("  Params:")
    for key, value in best.params.items():
        print(f"    {key}: {value}")

    test_scores = [t.value for t in study.trials]
    train_scores = [t.user_attrs["train_f1"] for t in study.trials]

    plt.figure(figsize=(12, 6))
    plt.plot(test_scores, label="Test data", color="blue")
    plt.plot(train_scores, label="Training data", color="red")
    plt.xlabel("Trial")
    plt.ylabel("F1 score")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()

    return best.params


if __name__ == "__main__":
    data_path = "data/created_dataset/train_df.pkl"
    optuna_optimization_history_path = "optuna_optimization_history_doc2vec.png"
    best_params_path = "optuna_doc2vec_best_params.pkl"

    train_df = load_pickle(data_path)
    best_params = optimize_doc2vec(
        train_df=train_df,
        categories=CATEGORIES,
        n_trials=500,
        output_path=optuna_optimization_history_path,
    )
    save_pickle(best_params, best_params_path)
