import pickle
from typing import List

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from rdkit.Chem import AllChem


# ---------------------------------------------------------------------------
# FpDoc2Vec
# ---------------------------------------------------------------------------

def _sum_doc_vectors(on_bits_list: List[List[int]], model: Doc2Vec) -> np.ndarray:
    """Sum Doc2Vec document vectors at each fingerprint's on-bit positions."""
    vectors = []
    for bits in on_bits_list:
        vec = np.zeros(model.vector_size)
        for b in bits:
            vec += model.dv.vectors[b]
        vectors.append(vec)
    return np.array(vectors)


def make_fp2vector(model_path: str, df: pd.DataFrame) -> np.ndarray:
    """Vectorize compounds using a pre-trained FpDoc2Vec model.

    Each compound vector is the sum of Doc2Vec document vectors at the
    on-bit positions of its ECFP fingerprint (column ``fp_3_4096``).

    Args:
        model_path: Path to the saved FpDoc2Vec ``.model`` file.
        df: DataFrame with a ``fp_3_4096`` column containing on-bit index lists.

    Returns:
        2-D array of shape ``(n_compounds, vector_size)``.
    """
    model = Doc2Vec.load(model_path)
    on_bits_list = list(df["fp_3_4096"])
    return _sum_doc_vectors(on_bits_list, model)


# ---------------------------------------------------------------------------
# NameDoc2Vec
# ---------------------------------------------------------------------------

def make_name2vector(model_path: str, df: pd.DataFrame) -> np.ndarray:
    """Vectorize compounds using a pre-trained NameDoc2Vec model.

    Each compound's vector is taken directly from the Doc2Vec document
    vectors by its row index in ``df``.

    Args:
        model_path: Path to the saved NameDoc2Vec ``.model`` file.
        df: DataFrame of compounds (only ``len(df)`` is used for indexing).

    Returns:
        2-D array of shape ``(n_compounds, vector_size)``.
    """
    model = Doc2Vec.load(model_path)
    return np.array([model.dv.vectors[i] for i in range(len(df))])


# ---------------------------------------------------------------------------
# ECFP (Morgan fingerprints)
# ---------------------------------------------------------------------------

def make_ecfp(df: pd.DataFrame, radius: int = 3, n_bits: int = 4096) -> np.ndarray:
    """Generate Morgan (ECFP) fingerprints for all compounds.

    Args:
        df: DataFrame with an ``ROMol`` column containing RDKit molecule objects.
        radius: Morgan fingerprint radius.
        n_bits: Fingerprint bit-vector length.

    Returns:
        2-D binary array of shape ``(n_compounds, n_bits)``.
        Molecules that fail fingerprint generation are skipped with a warning.
    """
    fingerprints = []
    for i, mol in enumerate(df["ROMol"]):
        try:
            fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits))
            fingerprints.append(fp)
        except Exception:
            print(f"Warning: could not generate fingerprint for molecule at index {i} — skipped.")
    return np.array(fingerprints)


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------

def make_descriptor(descriptor_path: str, feature_start_col: int = 14) -> np.ndarray:
    """Load pre-computed physicochemical descriptors from a pickled DataFrame.

    Args:
        descriptor_path: Path to the pickle file containing the descriptor DataFrame.
        feature_start_col: Column index where descriptor features start (default: 14).

    Returns:
        2-D array of shape ``(n_compounds, n_descriptors)``.
    """
    with open(descriptor_path, "rb") as f:
        df = pickle.load(f)
    return np.array(df.iloc[:, feature_start_col:])
