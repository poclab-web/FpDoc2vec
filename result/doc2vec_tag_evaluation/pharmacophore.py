
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from tqdm import tqdm
import pandas as pd


def process_pharmacophore_features(df: pd.DataFrame) -> Tuple[List[Optional[List[int]]], List[int]]:
    """
    Process pharmacophore features and identify invalid entries
    
    Args:
        df: DataFrame containing RDKit molecule objects in a column named 'ROMol'
        
    Returns:
        Tuple containing:
            - List of pharmacophore fingerprint bit indices (List[int]) or None for invalid entries
            - List of indices where pharmacophore generation failed
    """
    pharmacore_list = []
    for i in tqdm(df["ROMol"]):
        try:
            fp = Generate.Gen2DFingerprint(i, Gobbi_Pharm2D.factory)
            fp_bits = list(fp.GetOnBits())
            if len(fp_bits) == 0:
                pharmacore_list.append(None)
            else:
                pharmacore_list.append(fp_bits)
        except:
            print("Error")
            pharmacore_list.append(None)
            
    invalid_pharmacore_indices = []
    
    for idx, feature in enumerate(pharmacore_list):
        if feature is None:
            invalid_pharmacore_indices.append(idx)
            
    return pharmacore_list, invalid_pharmacore_indices
