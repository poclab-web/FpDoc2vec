from typing import List, Tuple
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
import pandas as pd
from gensim.models import Doc2Vec

def generate_ecfp_fingerprints(mols, radius: int, n_bits: int,) -> Tuple[np.ndarray, List[List[int]]]:
    """Generate ECFP fingerprints for a sequence of RDKit Mol objects."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp_array = []
    on_bits_list = []
    for i, mol in enumerate(mols):
        try:
            fp = generator.GetFingerprint(mol)
            on_bits_list.append(list(fp.GetOnBits()))
            fp_array.append(generator.GetFingerprintAsNumPy(mol))
        except Exception as e:
            print(f"Error processing molecule {i}: {e}")
    return np.array(fp_array), on_bits_list

def fingerprints_to_vectors(on_bits_list: List[List[int]], model: Doc2Vec) -> np.ndarray:
    """Create compound vectors by averaging Doc2Vec vectors at each on-bit position."""
    vectors = []
    for bits in on_bits_list:
        if len(bits) == 0:
            vectors.append(np.zeros(model.vector_size))
        else:
            vec = np.sum([model.dv.vectors[b] for b in bits], axis=0)
            vectors.append(vec)
    return np.array(vectors)

def make_name2vector(model: Doc2Vec, df: pd.DataFrame) -> np.ndarray:
    """Vectorize compounds using a pre-trained NameDoc2Vec model."""
    return np.array([model.dv.vectors[i] for i in range(len(df))])

def make_descriptor(df: pd.DataFrame) -> np.ndarray:
    """Load pre-computed physicochemical descriptors from a pickled DataFrame."""
    return np.array(df.select_dtypes(include=[np.floating]))


