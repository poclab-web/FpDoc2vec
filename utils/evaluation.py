from typing import Dict, List, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, matthews_corrcoef, balanced_accuracy_score,
    roc_auc_score, cohen_kappa_score, auc, precision_recall_curve,
)

from .constants import CATEGORIES, METRIC_NAMES


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Compute all classification metrics for a single prediction."""
    metrics = {}
    metrics['f1'] = f1_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    metrics['pr_auc'] = auc(recall, precision)
    return metrics


def _fit_and_score(classifier, X_train, X_test, y_train, y_test):
    """Fit the model on training data and compute metrics for both train and test sets."""
    classifier.fit(X_train, y_train)
    y_train_pred  = classifier.predict(X_train)
    y_test_pred   = classifier.predict(X_test)
    y_train_proba = classifier.predict_proba(X_train)[:, 1]
    y_test_proba  = classifier.predict_proba(X_test)[:, 1]
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
    test_metrics  = calculate_metrics(y_test,  y_test_pred,  y_test_proba)
    return train_metrics, test_metrics

def evaluate_category_cv(X_vec: np.ndarray,
                        y: np.ndarray,
                        classifier
                        ) -> Dict[str, Union[List[float], float]]:
    """Evaluate a single category with 5-fold stratified cross-validation."""

    all_train_scores = {m: [] for m in METRIC_NAMES}
    all_test_scores  = {m: [] for m in METRIC_NAMES}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_idx, test_idx in skf.split(range(len(y)), y):
        X_train, X_test = X_vec[train_idx], X_vec[test_idx]
        y_train, y_test = y[train_idx],     y[test_idx]

        train_metrics, test_metrics = _fit_and_score(classifier, X_train, X_test, y_train, y_test)
        for m in METRIC_NAMES:
            all_train_scores[m].append(train_metrics[m])
            all_test_scores[m].append(test_metrics[m])

    return {m: {
            'train_scores': all_train_scores[m],
            'test_scores':  all_test_scores[m],
            'mean_train':   np.mean(all_train_scores[m]),
            'mean_test':    np.mean(all_test_scores[m])}
        for m in METRIC_NAMES}

def main_cv(df: pd.DataFrame,
            X_vec: np.ndarray,
            classifier) -> Dict[str, Dict[str, float]]:
    """Run cross-validation evaluation across all categories."""
    results = {}
    for category in CATEGORIES:
        y = (df[category] == category).astype(int).to_numpy()
        results[category] = evaluate_category_cv(category, X_vec, y, classifier)
    return results


def evaluate_category_traintest(X_train_vec: np.ndarray,
                                X_test_vec: np.ndarray,
                                y_train: np.ndarray,
                                y_test: np.ndarray,
                                classifier
                                ) -> Dict[str, Union[List[float], float]]:
    """Evaluate a single category on a fixed train/test split."""

    train_metrics, test_metrics = _fit_and_score(classifier, X_train_vec, X_test_vec, y_train, y_test)

    return {m: {'train_scores': train_metrics[m],'test_scores':  test_metrics[m]} for m in METRIC_NAMES}


def main_traintest(train_df: pd.DataFrame,
                test_df: pd.DataFrame,
                X_train_vec: np.ndarray,
                X_test_vec: np.ndarray,
                classifier) -> Dict[str, Dict[str, float]]:
    
    """Run train/test evaluation across all categories."""
    results = {}
    for category in CATEGORIES:
        y_train = (train_df[category] == category).astype(int).to_numpy()
        y_test  = (test_df[category] == category).astype(int).to_numpy()
        results[category] = evaluate_category_traintest(
            X_train_vec, X_test_vec, y_train, y_test, classifier
        )
    return results
