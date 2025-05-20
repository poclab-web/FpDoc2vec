import pickle
import numpy as np
import pandas as pd
from typing import Union, Optional


def make_descriptor(input_path: str) -> np.ndarray:
    """Generate descriptor vectors from a pickled DataFrame
    
    Args:
        input_path: Path to the pickle file containing the DataFrame
        
    Returns:
        NumPy array of descriptor vectors extracted from columns 14 onwards
    """
    # Load dataset
    with open(input_path, "rb") as f:
        df = pickle.load(f)
    # Generate compound vectors
    desc = np.array(df.iloc[:, 14:])
    return desc
    
