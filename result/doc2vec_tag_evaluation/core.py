import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from gensim.models import Doc2Vec

from utils import (
    _fit_and_score,
    fingerprints_to_vectors,
    main_cv,
    CATEGORIES,
    METRIC_NAMES,
)

warnings.filterwarnings('ignore', message='X does not have valid feature names')

def evaluate_all_categories_filtered(
    X: np.ndarray,
    df: pd.DataFrame,
    df_filtered: pd.DataFrame,
    classifier,
    index_mapping: Dict[int, int],
) -> Dict:
    """Evaluate all categories with CV splits derived from the full df but features from df_filtered."""
    results = {}
    for category in CATEGORIES:
        y_full = (df[category] == category).astype(int).to_numpy()
        y_filtered = (df_filtered[category] == category).astype(int).to_numpy()

        all_train_scores = {m: [] for m in METRIC_NAMES}
        all_test_scores = {m: [] for m in METRIC_NAMES}

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for train_idx, test_idx in skf.split(range(len(y_full)), y_full):
            train_filtered = [index_mapping[i] for i in train_idx if i in index_mapping]
            test_filtered = [index_mapping[i] for i in test_idx if i in index_mapping]

            X_train, X_test = X[train_filtered], X[test_filtered]
            y_train, y_test = y_filtered[train_filtered], y_filtered[test_filtered]

            train_metrics, test_metrics = _fit_and_score(classifier, X_train, X_test, y_train, y_test)
            for m in METRIC_NAMES:
                all_train_scores[m].append(train_metrics[m])
                all_test_scores[m].append(test_metrics[m])

        results[category] = {
            m: {
                'train_scores': all_train_scores[m],
                'test_scores': all_test_scores[m],
                'mean_train': np.mean(all_train_scores[m]),
                'mean_test': np.mean(all_test_scores[m]),
            }
            for m in METRIC_NAMES
        }
    return results


def create_index_mapping(df_length: int, invalid_indices: List[int]) -> Dict[int, int]:
    """Map original DataFrame indices to filtered DataFrame indices (excluding invalid molecules)."""
    mapping = {}
    filtered_idx = 0
    invalid_set = set(invalid_indices)
    for orig_idx in range(df_length):
        if orig_idx not in invalid_set:
            mapping[orig_idx] = filtered_idx
            filtered_idx += 1
    return mapping


def run_with_filter(
    df: pd.DataFrame,
    on_bits_list: List[Optional[List[int]]],
    invalid_indices: List[int],
    classifier,
    model: Doc2Vec,
) -> Dict:
    """Evaluate when some molecules have invalid fingerprints."""
    valid_mask = np.array([b is not None for b in on_bits_list])
    df_filtered = df[valid_mask].reset_index(drop=True)
    valid_on_bits = [b for b in on_bits_list if b is not None]
    print(f"Compounds with valid fingerprints: {len(df_filtered)} / {len(df)}")
    X = fingerprints_to_vectors(valid_on_bits, model)

    index_mapping = create_index_mapping(len(df), invalid_indices)
    return evaluate_all_categories_filtered(X, df, df_filtered, classifier, index_mapping)


def run_ecfp(df: pd.DataFrame, on_bits_list: List[List[int]], classifier, model: Doc2Vec) -> Dict:
    """Evaluate FpDoc2Vec with ECFP tags via cross-validation."""
    X = fingerprints_to_vectors(on_bits_list, model)
    return main_cv(df, X, classifier)


def generate_maccs_on_bits(df: pd.DataFrame,) -> Tuple[List[Optional[List[int]]], List[int]]:
    """Generate MACCS key fingerprints as on-bit index lists for use as Doc2Vec tags."""
    on_bits_list = []
    invalid_indices = []

    for idx, mol in enumerate(df["ROMol"]):
        fp = MACCSkeys.GenMACCSKeys(mol)
        bits = list(fp.GetOnBits())
        if len(bits) == 0:
            print(f"No MACCS bits for molecule at index {idx}")
            on_bits_list.append(None)
            invalid_indices.append(idx)
        else:
            on_bits_list.append(bits)

    return on_bits_list, invalid_indices


def generate_pharmacophore_on_bits(
    df: pd.DataFrame,
) -> Tuple[List[Optional[List[int]]], List[int]]:
    """Generate 2D pharmacophore (Gobbi) fingerprints as on-bit index lists for use as Doc2Vec tags."""
    on_bits_list = []
    invalid_indices = []

    for idx, mol in enumerate(tqdm(df["ROMol"], desc="Generating pharmacophore fingerprints")):
        try:
            fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
            bits = list(fp.GetOnBits())
            if len(bits) == 0:
                on_bits_list.append(None)
                invalid_indices.append(idx)
            else:
                on_bits_list.append(bits)
        except Exception:
            print(f"Error processing molecule at index {idx}")
            on_bits_list.append(None)
            invalid_indices.append(idx)

    return on_bits_list, invalid_indices


def generate_ngram_indices(
    df: pd.DataFrame, smiles_col: str = "smiles", n: int = 3
) -> List[List[int]]:
    """Generate vocabulary-index lists from SMILES character n-grams for use as Doc2Vec tags."""
    smiles_list = list(df[smiles_col])
    ngrams = [
        [s] if len(s) < n else [s[i:i + n] for i in range(len(s) - n + 1)]
        for s in smiles_list
    ]
    vectorizer = CountVectorizer(binary=True, analyzer=lambda x: x)
    vec = vectorizer.fit_transform(ngrams)
    return [
        [j for j, val in enumerate(row) if val == 1]
        for row in vec.toarray()
    ]

