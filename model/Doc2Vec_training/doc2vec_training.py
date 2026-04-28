import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from rdkit.Chem import AllChem


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from config.doc2vec_config import (
    DESCRIPTION_COLUMN,
    DOC2VEC_PARAMS,
    FINGERPRINT_BITS,
    FINGERPRINT_COLUMN,
    FINGERPRINT_RADIUS,
)


def generate_morgan_fingerprints(
    df: pd.DataFrame, radius: int = FINGERPRINT_RADIUS, n_bits: int = FINGERPRINT_BITS
) -> List[List[int]]:
    """Return the active bit indices of Morgan fingerprints for each molecule.

    Args:
        df: DataFrame with an 'ROMol' column containing RDKit Mol objects.
        radius: Morgan fingerprint radius.
        n_bits: Fingerprint bit-vector length.

    Returns:
        List of lists, where each inner list contains the indices of bits set
        to 1 for that molecule's fingerprint.
    """
    fingerprints = []
    for i, mol in enumerate(df["ROMol"]):
        try:
            bits = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits))
            fingerprints.append(bits)
        except Exception:
            print(f"Warning: skipping molecule at index {i} (fingerprint error).")

    fp_array = np.array(fingerprints)
    return [[j for j in range(n_bits) if row[j] == 1] for row in fp_array]


def lowercasing(x: Any) -> Any:
    """Recursively convert strings (or collections of strings) to lowercase.

    Args:
        x: A str, list, tuple, or value convertible via str().

    Returns:
        Lowercase version of the input with the same structure.
    """
    if isinstance(x, (list, tuple)):
        return [lowercasing(item) for item in x]
    if isinstance(x, str):
        return x.lower()
    try:
        return str(x).lower()
    except Exception as e:
        raise TypeError(f"Cannot lowercase value of type {type(x)}") from e


def extract_compound_names(df: pd.DataFrame) -> List[str]:
    """Extract and lowercase the first compound name from each row.

    Args:
        df: DataFrame with a 'compounds' column where each entry is a list
            whose first element is the compound name.

    Returns:
        List of lowercase compound names.
    """
    names = [row[0] for row in df["compounds"]]
    return lowercasing(names)


def build_tagged_corpus(df: pd.DataFrame, tag_list: List[Any], description_column: str = DESCRIPTION_COLUMN,) -> List[TaggedDocument]:
    """Convert a DataFrame of preprocessed text into a tagged corpus.

    Args:
        df: DataFrame with a column of tokenised, nested lists of words.
        tag_list: Tags for each document (e.g. fingerprint bit indices or compound name strings). Must match the row count of df.
        description_column: Column in df that holds the preprocessed text.

    Returns:
        List of TaggedDocument objects ready for Doc2Vec training.
    """
    tagged = []
    for i, doc in enumerate(df[description_column]):
        words = sum(doc, [])  # flatten list-of-lists → flat word list
        tagged.append(TaggedDocument(words=words, tags=[tag_list[i]]))
    return tagged


def train_doc2vec(df: pd.DataFrame, tag_list: List[Any], params: Dict[str, Any] = DOC2VEC_PARAMS, description_column: str = DESCRIPTION_COLUMN,) -> Doc2Vec:
    """Train a Doc2Vec model.

    Args:
        df: DataFrame containing preprocessed descriptions.
        tag_list: Per-document tags (fingerprint bits or compound names).
        params: Doc2Vec constructor keyword arguments.
        description_column: Column in df with the tokenised text.

    Returns:
        Trained Doc2Vec model.
    """
    corpus = build_tagged_corpus(df, tag_list, description_column)
    return Doc2Vec(corpus, **params)


def train_and_save(
    dataset_path: Path,
    output_path: Path,
    tag_list: List[Any],
    params: Dict[str, Any] = DOC2VEC_PARAMS,
    description_column: str = DESCRIPTION_COLUMN,
) -> None:
    """Load a dataset, train a Doc2Vec model, and save it to disk.

    Args:
        dataset_path: Path to a pickle file containing a pandas DataFrame.
        output_path: Where to save the trained model.
        tag_list: Per-document tags (fingerprint bits or compound names).
        params: Doc2Vec constructor keyword arguments.
        description_column: Column in the DataFrame with preprocessed text.
    """
    with open(dataset_path, "rb") as f:
        df = pickle.load(f)

    model = train_doc2vec(df, tag_list, params, description_column)
    model.save(str(output_path))



if __name__ == "__main__":
    # Example usage - replace with your actual file paths

    # --- Load the training dataset once for shared use ---
    with open("train_df.pkl", "rb") as f:
        df_full = pickle.load(f)

    # 1. FpDoc2Vec (training dataset, Morgan fingerprint tags)
    fp_tags_full = list(df_full[FINGERPRINT_COLUMN])
    train_and_save("train_df.pkl", "fpdoc2vec.model", fp_tags_full)

    # 2. NameDoc2Vec (training dataset, compound name tags)
    name_tags = extract_compound_names(df_full)
    train_and_save("train_df.pkl", "namedoc2vec.model", name_tags)
