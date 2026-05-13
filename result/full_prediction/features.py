import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from utils import fingerprints_to_vectors


# ---------------------------------------------------------------------------
# FpDoc2Vec
# ---------------------------------------------------------------------------

def make_fp2vector(model: Doc2Vec, df: pd.DataFrame) -> np.ndarray:

    finger_list = list(df["fp_3_4096"])
    compound_vec = fingerprints_to_vectors(finger_list, model)
    vec = np.array(compound_vec)
    return vec


# ---------------------------------------------------------------------------
# NameDoc2Vec
# ---------------------------------------------------------------------------

def make_name2vector(model: Doc2Vec, df: pd.DataFrame) -> np.ndarray:
    """Vectorize compounds using a pre-trained NameDoc2Vec model.

    Each compound's vector is taken directly from the Doc2Vec document
    vectors by its row index in ``df``.

    Args:
        model: Pre-trained NameDoc2Vec model.
        df: DataFrame of compounds (only ``len(df)`` is used for indexing).

    Returns:
        2-D array of shape ``(n_compounds, vector_size)``.
    """
    return np.array([model.dv.vectors[i] for i in range(len(df))])


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------

def make_descriptor(df: pd.DataFrame) -> np.ndarray:
    """Load pre-computed physicochemical descriptors from a pickled DataFrame.

    Args:
        df: DataFrame containing the descriptors.

    Returns:
        2-D array of shape ``(n_compounds, n_descriptors)``.
    """
    return np.array(df.select_dtypes(include=[np.floating]))