import numpy as np
import pickle
from typing import List, Dict, Tuple, Union, Optional, Any
from numpy.linalg import norm
from gensim.models.doc2vec import Doc2Vec
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import AllChem

from similarity_namedoc2vec import find_similar_terms
from similarity_fpdoc2vec import cosine_similarity, calculate_similarities, get_top_similar_compounds, add_vectors
from similarity_tanimoto import calculate_tanimoto_similarities, get_top_similar_compounds


def similarity_fpdoc2vec(input_path: str, model_path: str, target_compound: str = "sucrose", n: int = 10) -> None:
    """
    Load data and model, then find compounds similar to the target compound.
    
    Args:
        input_path: Path to the pickle file containing the dataset
        model_path: Path to the FpDoc2Vec model file
        target_compound: Name of the target compound (default: "sucrose")
        n: Number of similar compounds to return (default: 10)
        
    Returns:
        None
    """
    # Load dataset
    with open(input_path, "rb") as f:
        df = pickle.load(f)
    
    # Add compound names as a separate column
    df["NAME"] = [df.iat[i, 0][0] for i in range(len(df))]
    
    # Get fingerprints and load model
    finger_list = list(df["fp_3_4096"])
    model = Doc2Vec.load(model_path)
    
    # Generate compound vectors
    compound_vec = add_vectors(finger_list, model)
    
    # Calculate similarities to sucrose
    df[target_compound] = calculate_similarities(df, compound_vec, target_compound)
    
    # Get top n similar compounds
    top_similar = get_top_similar_compounds(df, target_compound, n)
    
    # Display results in the same format as the previous code
    print(f"・Similar terms to '{target_compound}' with similarity scores:")
    for idx, row in top_similar.iterrows():
        compound_name = row['NAME']
        similarity_score = row[target_compound]
        print(f"  {compound_name}: {similarity_score:.4f}")


def similarity_namedoc2vec(input_path: str, model_path: str, target_compound: str = "sucrose", n: int = 10) -> None:
    """
    Load data and model, then demonstrate similarity search.
    
    Args:
        input_path: Path to the pickle file containing the dataset
        model_path: Path to the trained NameDoc2Vec model file
        target_compound: Name of the target compound (default: "sucrose")
        n: Number of similar compounds to return (default: 10)
        
    Returns:
        None
    """
    # Load dataset
    with open(input_path, "rb") as f:
        df = pickle.load(f)
    
    # Load pre-trained Doc2Vec model
    model = Doc2Vec.load(model_path)
    
    # Find terms similar to target compound
    similar_terms = find_similar_terms(model, target_compound, n)
    
    # Display results
    print(f"・Similar terms to '{target_compound}' with similarity scores:")
    for term, score in similar_terms:
        print(f"  {term}: {score:.4f}")


def similarity_tanimoto(input_path: str, target_compound: str = "sucrose", n: int = 10) -> None:
    """
    Load data and calculate Tanimoto similarities to the target compound.
    
    Args:
        input_path: Path to the pickle file containing the compound DataFrame
        target_compound: Name of the target compound to compare against (default: "sucrose")
        n: Number of similar compounds to return (default: 10)
        
    Returns:
        None
    """
    # Load dataset
    with open(input_path, "rb") as f:
        df = pickle.load(f)
    
    # Add compound names as a separate column if not already present
    if "NAME" not in df.columns:
        df["NAME"] = [df.iat[i, 0][0] for i in range(len(df))]
    
    # Calculate Tanimoto similarities to target compound
    tanimoto_similarities = calculate_tanimoto_similarities(df, target_compound)
    
    # Get top n similar compounds
    top_similar = get_top_similar_compounds(df, tanimoto_similarities, target_compound, n)
    
    # Display results in the same format as before
    print(f"・Similar terms to '{target_compound}' with similarity scores:")
    for idx, row in top_similar.iterrows():
        compound_name = row['NAME']
        similarity_score = row["tanimoto"]
        print(f"  {compound_name}: {similarity_score:.4f}")


# Example usage - replace with your actual file paths
if __name__ == "__main__":
    input_path = "10genre_dataset.pkl"
    n = 10  # Please change the values as you like.
  
    # Compound Similarity Using FpDoc2Vec Model
    model_path_fp = "fpdoc2vec.model"
    similarity_fpdoc2vec(input_path, model_path_fp, n=n)
  
    # Compound Similarity Using NameDoc2Vec Model
    model_path_name = "namedoc2vec.model"
    similarity_namedoc2vec(input_path, model_path_name, n=n)

    # Compound Similarity Using Tanimoto similarity
    similarity_tanimoto(input_path, n=n)
