
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
from rdkit.Chem import rdFingerprintGenerator


def generate_ecfp_fingerprints(
    df: pd.DataFrame,
    radius: int,
    n_bits: int,
) -> Tuple[np.ndarray, List[List[int]]]:
    """Generate ECFP fingerprints for all molecules in a DataFrame.

    Uses RDKit's MorganGenerator, which is the recommended API over the
    deprecated GetMorganFingerprintAsBitVect.

    Args:
        df: DataFrame with an 'ROMol' column containing RDKit Mol objects
        radius: Morgan fingerprint radius (e.g. 2 or 3)
        n_bits: Fingerprint bit-vector length (e.g. 2048, 4096, 8192)

    Returns:
        Tuple of:
          - fp_array: numpy array of shape (n_molecules, n_bits)
          - on_bits_list: list of on-bit index lists, one list per molecule;
                          used as Doc2Vec document tags in FpDoc2Vec
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp_array = []
    on_bits_list = []
    for i, mol in enumerate(df["ROMol"]):
        try:
            fp = generator.GetFingerprint(mol)
            on_bits_list.append(list(fp.GetOnBits()))
            fp_array.append(generator.GetFingerprintAsNumPy(mol))
        except Exception as e:
            print(f"Error processing molecule {i}: {e}")
    return np.array(fp_array), on_bits_list




def load_descriptors(
    descriptor_path: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load pre-computed molecular descriptors and align with train/test split.

    The descriptor file must be a pickle containing a DataFrame with an 'inchikey'
    column for matching and floating-point descriptor columns (all numeric columns
    are used as features).

    Args:
        descriptor_path: Path to the pickle file containing the descriptor DataFrame
        train_df: Training DataFrame with an 'inchikey' column
        test_df: Test DataFrame with an 'inchikey' column

    Returns:
        Tuple of (train_df_desc, test_df_desc, X_train, X_test) where:
          - train_df_desc / test_df_desc: descriptor DataFrame rows for each split
          - X_train / X_test: 2D numpy arrays of descriptor values
    """
    with open(descriptor_path, "rb") as f:
        df = pickle.load(f)

    train_df_desc = df[df["inchikey"].isin(list(train_df["inchikey"]))]
    test_df_desc = df[df["inchikey"].isin(list(test_df["inchikey"]))]

    # Select only floating-point columns (descriptor values, not metadata)
    X_train = np.array(train_df_desc.select_dtypes(include=[np.floating]))
    X_test = np.array(test_df_desc.select_dtypes(include=[np.floating]))

    return train_df_desc, test_df_desc, X_train, X_test
