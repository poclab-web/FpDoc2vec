import os
import pickle
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit.Chem import MACCSkeys, Generate
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from config.doc2vec_params import doc2vec_param as DOC2VEC_PARAMS
from util import (
    build_doc2vec_model,
    fingerprints_to_vectors,
    generate_ecfp_fingerprints,
    evaluate_all_categories,
    evaluate_all_categories_filtered,
    print_mcc_summary,
)

warnings.filterwarnings('ignore', message='X does not have valid feature names')

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

DATA_PATH = "../../data/created_dataset/train_df.pkl"
DESCRIPTION_COL = "description_gensim"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_results(results: Dict, filename: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, filename), "wb") as f:
        pickle.dump(results, f)


def run_ecfp(
    df: pd.DataFrame,
    radius: int,
    n_bits: int,
    lgbm: lgb.LGBMClassifier,
    desc_col: str,
) -> Dict:
    """Train and evaluate Doc2Vec + LightGBM using ECFP on-bits as document tags."""
    _, on_bits = generate_ecfp_fingerprints(df, radius, n_bits)
    model = build_doc2vec_model(df[desc_col].tolist(), on_bits, DOC2VEC_PARAMS)
    X = fingerprints_to_vectors(on_bits, model)
    return evaluate_all_categories(X, df, lgbm)


def run_with_filter(
    df: pd.DataFrame,
    on_bits_list: List[Optional[List[int]]],
    invalid_indices: List[int],
    lgbm: lgb.LGBMClassifier,
    desc_col: str,
) -> Dict:
    """Train and evaluate when some molecules have invalid fingerprints.

    The filtered subset is used for Doc2Vec training and vectorization.
    Cross-validation splits are derived from the full dataset for consistency.
    """
    valid_mask = np.array([b is not None for b in on_bits_list])
    df_filtered = df[valid_mask].reset_index(drop=True)
    valid_on_bits = [b for b in on_bits_list if b is not None]
    print(f"Compounds with valid fingerprints: {len(df_filtered)} / {len(df)}")

    model = build_doc2vec_model(df_filtered[desc_col].tolist(), valid_on_bits, DOC2VEC_PARAMS)
    X = fingerprints_to_vectors(valid_on_bits, model)

    index_mapping = create_index_mapping(len(df), invalid_indices)
    return evaluate_all_categories_filtered(X, df, df_filtered, lgbm, index_mapping)

def generate_maccs_on_bits(
    df: pd.DataFrame,
) -> Tuple[List[Optional[List[int]]], List[int]]:
    """Generate MACCS key fingerprints as on-bit index lists for use as Doc2Vec tags.

    Args:
        df: DataFrame with RDKit Mol objects in the 'ROMol' column

    Returns:
        Tuple of:
          - on_bits_list: on-bit index lists per molecule (None for invalid molecules)
          - invalid_indices: indices of molecules with no bits set
    """
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
    """Generate 2D pharmacophore (Gobbi) fingerprints as on-bit index lists for use as Doc2Vec tags.

    Args:
        df: DataFrame with RDKit Mol objects in the 'ROMol' column

    Returns:
        Tuple of:
          - on_bits_list: on-bit index lists per molecule (None for invalid molecules)
          - invalid_indices: indices of molecules where generation failed or no bits are set
    """
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
    """Generate vocabulary-index lists from SMILES character n-grams for use as Doc2Vec tags.

    Builds a shared n-gram vocabulary across all molecules and represents each
    molecule as the indices of its present n-grams.

    Args:
        df: DataFrame with a column containing SMILES strings
        smiles_col: Name of the SMILES column
        n: n-gram window size

    Returns:
        List of index lists, one per molecule
    """
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


def create_index_mapping(df_length: int, invalid_indices: List[int]) -> Dict[int, int]:
    """Map original DataFrame indices to filtered DataFrame indices (excluding invalid molecules).

    Args:
        df_length: Length of the original DataFrame
        invalid_indices: Indices of molecules to exclude

    Returns:
        Dict mapping original index -> filtered index
    """
    mapping = {}
    filtered_idx = 0
    invalid_set = set(invalid_indices)
    for orig_idx in range(df_length):
        if orig_idx not in invalid_set:
            mapping[orig_idx] = filtered_idx
            filtered_idx += 1
    return mapping
