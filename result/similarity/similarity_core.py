import pandas as pd
import numpy as np
from numpy.linalg import norm
from IPython.display import display
from typing import List


from utils import CATEGORIES

# Fp Doc2vec, Name Doc2vec

def get_categories(df: pd.DataFrame, idx: int) -> str:
    
    found_categories = []
    for cat in CATEGORIES:
        if cat in df.columns and df.iat[idx, df.columns.get_loc(cat)] != 'No':
            found_categories.append(cat)
    return ', '.join(found_categories)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def calculate_similarities(df: pd.DataFrame, vectors: np.ndarray, target_compound_name: str) -> List[float]:
    try:
        
        target_indices = df[df["NAME"] == target_compound_name].index
        target_idx = target_indices[0]
        target_vector = vectors[target_idx]
        similarities = []
        for i, vector in enumerate(vectors):
            if i != target_idx:
                sim = cosine_similarity(vector, target_vector)
                similarities.append(sim)
            else:
                similarities.append(0)       
        return similarities
        
    except IndexError:
        raise ValueError(f"Compound '{target_compound_name}' not found in the dataset")

def similarity_output(df, compound_vec:np.ndarray, target_compound: str, n: int) -> None:
    
    # Calculate similarities to sucrose
    df[target_compound] = calculate_similarities(df, compound_vec, target_compound)
    
    # Get top n similar compounds
    sorted_df = df.sort_values(target_compound, ascending=False)
    
    # Return the top N compounds (excluding the target compound itself)
    top_similar = sorted_df.head(n)[['NAME', target_compound]]
    target_idx = df[df["NAME"] == target_compound].index[0]
    target_categories = get_categories(df, target_idx)
    print(f"Target compound '{target_compound}' categories: {target_categories}")

    # Display results in the same format as the previous code
    print(f"Top {n} most similar compounds to '{target_compound}':")
    for idx, row in top_similar.iterrows():
        compound_name = row['NAME']
        similarity_score = row[target_compound]
        categories = get_categories(df, idx)  
        print(f"  {compound_name}: {similarity_score:.4f} (Categories: {categories})")  
        display(df.at[idx, 'ROMol'])




# ECFP

def calculate_tanimoto_similarities(df, fingerprint_array: np.ndarray, target_compound_name: str):
    target_idx = df[df["NAME"] == target_compound_name].index[0]
    target_fp = fingerprint_array[target_idx]
    
    dot = fingerprint_array @ target_fp                      
    similarities = dot / (fingerprint_array.sum(axis=1) + target_fp.sum() - dot)
    similarities[target_idx] = 0.0
    return similarities
    
def tanimoto_similarity_output(df: pd.DataFrame, fingerprint_objects: List, target_compound: str, n: int = 10) -> None:

    
    df[target_compound] = calculate_tanimoto_similarities(df, fingerprint_objects, target_compound)
    
    # Get top n similar compounds
    sorted_df = df.sort_values(target_compound, ascending=False)
    top_similar = sorted_df.head(n)[['NAME', target_compound]]
    target_idx = df[df["NAME"] == target_compound].index[0]
    target_categories = get_categories(df, target_idx)
    # Display results with 10th column
    print(f"Target compound '{target_compound}' categories: {target_categories}")
    print(f"Top {n} most similar compounds to '{target_compound}' (Tanimoto similarity):")
    for idx, row in top_similar.iterrows():
        compound_name = row['NAME']
        similarity_score = row[target_compound]
        categories = get_categories(df, idx)  
        print(f"  {compound_name}: {similarity_score:.4f} (Categories: {categories})")
        display(df.at[idx, 'ROMol'])