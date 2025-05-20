import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Callable, Optional, Any, Tuple
from rdkit import Chem
from rdkit.Chem import PandasTools, AllChem
from data_loading import mol_to_inchikey
from preprocessing import lowercasing


def add_property_column(df: pd.DataFrame, property_name: str, sdf_path: str) -> pd.DataFrame:
    """Add a property column based on InChIKey matching with compounds from an SDF file
    
    Args:
        df: DataFrame containing compound information with an 'inchikey' column
        property_name: Name of the property to add (will be used as column name)
        sdf_path: Path to the SDF file containing compounds with the specified property
        
    Returns:
        DataFrame with the new property column added
    """
    property_df = PandasTools.LoadSDF(sdf_path)
    property_df["inchikey"] = mol_to_inchikey(property_df)
    df[property_name] = [property_name if i in list(property_df['inchikey']) else "No" for i in df["inchikey"]]
    return df


def generate_morgan_fingerprints(df: pd.DataFrame, radius: int, n_bits: int) -> List[List[int]]:
    """
    Generate Morgan fingerprints for molecules in the dataframe.
    
    Args:
        df: DataFrame containing RDKit molecule objects in a column named 'ROMol'
        radius: The radius of the Morgan fingerprint. Higher values capture more extended 
                connectivity information around each atom
        n_bits: The length of the bit vector. Larger values reduce the chance of bit collisions
    
    Returns:
        A list containing the indices of bits set to 1 for each molecule's fingerprint.
        Each inner list represents the active bits for a single molecule.
    """
    fingerprints = []
    for i, mol in enumerate(df["ROMol"]):
        try:
            fingerprint = [j for j in AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)]
            fingerprints.append(fingerprint)
        except:
            print(f"Error processing molecule at index {i}")
            continue
    fingerprints = np.array(fingerprints)
    return [[j for j in range(n_bits) if i[j] == 1] for i in fingerprints]
