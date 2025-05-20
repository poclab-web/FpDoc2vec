import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec


def make_name2vector(model_path: str, df: pd.DataFrame) -> np.ndarray:
    """Convert to compound vectors using NameDoc2Vec model
    
    Args:
        model_path: Path to the saved NameDoc2Vec model file
        df: DataFrame containing compound data
        
    Returns:
        NumPy array of document vectors with shape (len(df), vector_size)
    """
    model = Doc2Vec.load(model_path)
    vec = np.array([model.dv.vectors[i] for i in range(len(df))])
    return vec
    
