
from typing import List, Tuple

import numpy as np
from rdkit.Chem import rdFingerprintGenerator


def generate_ecfp_fingerprints(
    mols,
    radius: int,
    n_bits: int,
) -> Tuple[np.ndarray, List[List[int]]]:
    """Generate ECFP fingerprints for a sequence of RDKit Mol objects.

    Uses RDKit's MorganGenerator, which is the recommended API over the
    deprecated GetMorganFingerprintAsBitVect.

    Args:
        mols: iterable of RDKit Mol objects (e.g. df["ROMol"])
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
    for i, mol in enumerate(mols):
        try:
            fp = generator.GetFingerprint(mol)
            on_bits_list.append(list(fp.GetOnBits()))
            fp_array.append(generator.GetFingerprintAsNumPy(mol))
        except Exception as e:
            print(f"Error processing molecule {i}: {e}")
    return np.array(fp_array), on_bits_list


