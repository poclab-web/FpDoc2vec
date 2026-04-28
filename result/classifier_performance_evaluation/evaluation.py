import pickle
from typing import Dict, List, Union

import numpy as np
from gensim.models.doc2vec import Doc2Vec
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


def fin(df, radius: int, fpSize: int):
    fingerprints = []
    onbits_list = []
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)
    for i, mol in enumerate(df["ROMol"]):
        try:
            fp = fp_generator.GetFingerprint(mol)
            onbits_list.append(list(fp.GetOnBits()))
            fingerprints.append(fp_generator.GetFingerprintAsNumPy(mol))
        except Exception as e:
            print(f"Error processing molecule {i}: {e}")
            continue
    return np.array(fingerprints), onbits_list


def add_vectors(fp_list: List[List[int]], model: Doc2Vec) -> List[np.ndarray]:
    compound_vec = []
    for i in fp_list:
        fingerprint_vec = 0
        for j in i:
            fingerprint_vec += model.dv.vectors[j]
        compound_vec.append(fingerprint_vec)
    return compound_vec


def calculate_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return {
        "f1": f1_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "pr_auc": auc(recall, precision),
    }


def evaluate_category(
    X_vec: np.ndarray,
    y: np.ndarray,
    classifier,
) -> Dict[str, Union[List[float], float]]:
    metric_names = ["f1", "mcc", "balanced_accuracy", "roc_auc", "kappa", "pr_auc"]
    all_train_scores = {m: [] for m in metric_names}
    all_test_scores = {m: [] for m in metric_names}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_idx, test_idx in skf.split(range(len(y)), y):
        X_train, X_test = X_vec[train_idx], X_vec[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        classifier.fit(X_train, y_train)
        train_metrics = calculate_metrics(y_train, classifier.predict(X_train), classifier.predict_proba(X_train)[:, 1])
        test_metrics = calculate_metrics(y_test, classifier.predict(X_test), classifier.predict_proba(X_test)[:, 1])

        for m in metric_names:
            all_train_scores[m].append(train_metrics[m])
            all_test_scores[m].append(test_metrics[m])

    return {
        m: {
            "train_scores": all_train_scores[m],
            "test_scores": all_test_scores[m],
            "mean_train": np.mean(all_train_scores[m]),
            "mean_test": np.mean(all_test_scores[m]),
        }
        for m in metric_names
    }


CATEGORIES = [
    "antioxidant",
    "anti_inflammatory_agent",
    "allergen",
    "dye",
    "toxin",
    "flavouring_agent",
    "agrochemical",
    "volatile_oil",
    "antibacterial_agent",
    "insecticide",
]


def run_evaluation(
    input_path: str,
    model_path: str,
    radius: int,
    fp_size: int,
    classifier,
) -> Dict[str, Dict[str, float]]:
    with open(input_path, "rb") as f:
        df = pickle.load(f)
    model = Doc2Vec.load(model_path)

    bit_list = fin(df, radius, fp_size)[1]
    X_vec = np.array(add_vectors(bit_list, model))

    results = {}
    for category in CATEGORIES:
        y = np.array([1 if i == category else 0 for i in df[category]])
        results[category] = evaluate_category(X_vec, y, classifier)
    return results
