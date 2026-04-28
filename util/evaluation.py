from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, matthews_corrcoef, balanced_accuracy_score,
    roc_auc_score, cohen_kappa_score, auc, precision_recall_curve,
)

from .constants import CATEGORIES, METRIC_NAMES


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Compute all classification metrics for a single prediction.

    Args:
        y_true: Ground-truth binary labels
        y_pred: Predicted binary labels
        y_proba: Predicted probabilities for the positive class

    Returns:
        Dict with keys: f1, mcc, balanced_accuracy, roc_auc, kappa, pr_auc
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return {
        'f1': f1_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'pr_auc': auc(recall, precision),
    }


# ---------------------------------------------------------------------------
# Train/test split mode (unseen-compound prediction)
# ---------------------------------------------------------------------------

def evaluate_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model,
) -> Dict[str, Dict]:
    """Fit model on training data and evaluate on both splits.

    Scores are wrapped in single-element lists so the result format matches
    the cross-validation output, making downstream aggregation uniform.

    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix
        y_train: Training binary labels
        y_test: Test binary labels
        model: Scikit-learn compatible classifier

    Returns:
        Dict[metric_name -> {'train_scores': [float], 'test_scores': [float]}]
    """
    model.fit(X_train, y_train)
    train_m = calculate_metrics(y_train, model.predict(X_train), model.predict_proba(X_train)[:, 1])
    test_m = calculate_metrics(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1])
    return {
        m: {'train_scores': [train_m[m]], 'test_scores': [test_m[m]]}
        for m in METRIC_NAMES
    }


def evaluate_all_categories_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model,
    categories: List[str] = CATEGORIES,
) -> Dict[str, Dict]:
    """Evaluate all categories using a pre-split train/test dataset.

    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix
        train_df: Training DataFrame; must contain one column per category
        test_df: Test DataFrame; must contain one column per category
        model: Scikit-learn compatible classifier
        categories: Category column names to evaluate

    Returns:
        Dict[category -> Dict[metric -> {'train_scores', 'test_scores'}]]
    """
    results = {}
    for cat in categories:
        y_train = np.array([1 if v == cat else 0 for v in train_df[cat]])
        y_test = np.array([1 if v == cat else 0 for v in test_df[cat]])
        results[cat] = evaluate_train_test(X_train, X_test, y_train, y_test, model)
    return results


# ---------------------------------------------------------------------------
# Cross-validation mode (tag comparison experiments)
# ---------------------------------------------------------------------------

def _run_cv(X: np.ndarray, y: np.ndarray, model) -> Dict[str, Dict]:
    """Run 5-fold stratified cross-validation for a single binary label array.

    Returns per-fold scores and fold-mean scores for every metric.
    """
    train_scores = {m: [] for m in METRIC_NAMES}
    test_scores = {m: [] for m in METRIC_NAMES}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_idx, test_idx in skf.split(range(len(y)), y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        train_m = calculate_metrics(y_train, model.predict(X_train), model.predict_proba(X_train)[:, 1])
        test_m = calculate_metrics(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1])

        for m in METRIC_NAMES:
            train_scores[m].append(train_m[m])
            test_scores[m].append(test_m[m])

    return {
        m: {
            'train_scores': train_scores[m],
            'test_scores': test_scores[m],
            'mean_train': np.mean(train_scores[m]),
            'mean_test': np.mean(test_scores[m]),
        }
        for m in METRIC_NAMES
    }


def evaluate_all_categories(
    X: np.ndarray,
    df: pd.DataFrame,
    model,
    categories: List[str] = CATEGORIES,
) -> Dict[str, Dict]:
    """Evaluate all categories with 5-fold stratified cross-validation.

    Args:
        X: Feature matrix for all compounds
        df: DataFrame containing category label columns
        model: Scikit-learn compatible classifier
        categories: Category column names to evaluate

    Returns:
        Dict[category -> Dict[metric -> {'train_scores', 'test_scores', 'mean_train', 'mean_test'}]]
    """
    return {
        cat: _run_cv(X, np.array([1 if v == cat else 0 for v in df[cat]]), model)
        for cat in categories
    }


def evaluate_all_categories_filtered(
    X: np.ndarray,
    df: pd.DataFrame,
    df_filtered: pd.DataFrame,
    model,
    index_mapping: Dict[int, int],
    categories: List[str] = CATEGORIES,
) -> Dict[str, Dict]:
    """Evaluate with 5-fold CV when some molecules have invalid fingerprints.

    CV splits are derived from the full dataset for reproducibility, then
    remapped to filtered-subset indices.

    Args:
        X: Feature matrix for the filtered (valid) compounds
        df: Full DataFrame used to generate consistent CV splits
        df_filtered: Filtered DataFrame matching rows of X
        model: Scikit-learn compatible classifier
        index_mapping: Dict mapping original indices to filtered indices
        categories: Category column names to evaluate

    Returns:
        Same structure as evaluate_all_categories
    """
    results = {}
    for cat in categories:
        y_all = np.array([1 if v == cat else 0 for v in df[cat]])
        y_filtered = np.array([1 if v == cat else 0 for v in df_filtered[cat]])

        train_scores = {m: [] for m in METRIC_NAMES}
        test_scores = {m: [] for m in METRIC_NAMES}

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for train_idx, test_idx in skf.split(range(len(y_all)), y_all):
            train_f = [index_mapping[i] for i in train_idx if i in index_mapping]
            test_f = [index_mapping[i] for i in test_idx if i in index_mapping]

            X_train, X_test = X[train_f], X[test_f]
            y_train, y_test = y_filtered[train_f], y_filtered[test_f]

            model.fit(X_train, y_train)
            train_m = calculate_metrics(y_train, model.predict(X_train), model.predict_proba(X_train)[:, 1])
            test_m = calculate_metrics(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1])

            for m in METRIC_NAMES:
                train_scores[m].append(train_m[m])
                test_scores[m].append(test_m[m])

        results[cat] = {
            m: {
                'train_scores': train_scores[m],
                'test_scores': test_scores[m],
                'mean_train': np.mean(train_scores[m]),
                'mean_test': np.mean(test_scores[m]),
            }
            for m in METRIC_NAMES
        }

    return results


# ---------------------------------------------------------------------------
# Result printing
# ---------------------------------------------------------------------------

def print_metric_summary(results: Dict[str, Dict], label: str, metric: str = 'mcc') -> None:
    """Print per-category scores and their mean for the specified metric.

    Works with both CV results (uses 'mean_test') and train/test results
    (uses the single value in 'test_scores').

    Args:
        results: Output of any evaluate_* function
        label: Header label for the summary block
        metric: Metric name to display (default: 'mcc')
    """
    print(f"\n=== {label} ===")
    scores = []
    for cat in CATEGORIES:
        if cat not in results:
            continue
        cat_result = results[cat][metric]
        # CV results have 'mean_test'; train/test results have a single-element list
        score = cat_result.get('mean_test', cat_result['test_scores'][0])
        print(f"  {cat}: {score:.4f}")
        scores.append(score)
    print(f"  Mean {metric.upper()}: {np.mean(scores):.4f}")


def print_mcc_summary(results: Dict[str, Dict], label: str) -> None:
    """Convenience wrapper that prints MCC summary (backward compatible)."""
    print_metric_summary(results, label, metric='mcc')
