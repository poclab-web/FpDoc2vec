import numpy as np
import pickle
from numpy.linalg import norm
from gensim.models.doc2vec import Doc2Vec
import pandas as pd

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score between the vectors
    """
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def calculate_similarities(df, vectors, target_compound):
    """
    Calculate cosine similarities between a target compound and all other compounds.
    
    Args:
        df: DataFrame containing compound information
        vectors: List of compound vectors
        target_compound: Name of the target compound
        
    Returns:
        List of cosine similarity scores
    """
    # Find the index of the target compound
    target_idx = None
    for i in range(len(df)):
        if df.iat[i, 22] == target_compound:
            target_idx = i
            break
    
    if target_idx is None:
        raise ValueError(f"Compound '{target_compound}' not found in the dataset")
    
    # Calculate cosine similarity for all compounds
    similarities = []
    for i in range(len(df)):
        if i != target_idx:
            similarity = np.dot(vectors[target_idx], vectors[i]) / (norm(vectors[target_idx]) * norm(vectors[i]))
            similarities.append(similarity)
        else:
            similarities.append(0)  # Self-similarity set to 0
    
    return similarities

def get_top_similar_compounds(df, target_compound, n=10):
    """
    Find the top N compounds most similar to the target compound.
    
    Args:
        df: DataFrame containing compound information and similarity scores
        target_compound: Name of the target compound
        n: Number of similar compounds to return (default: 10)
        
    Returns:
        DataFrame with the top N most similar compounds
    """
    # Sort by similarity score in descending order
    sorted_df = df.sort_values(target_compound, ascending=False)
    
    # Return the top N compounds (excluding the target compound itself)
    return sorted_df.head(n)[['NAME', target_compound]]

def main():
    """Load data and model, then find compounds similar to sucrose."""
    # Load dataset
    with open("../../data/10genre_dataset.pkl", "rb") as f:
        df = pickle.load(f)
    
    # Add compound names as a separate column
    df["NAME"] = [df.iat[i, 0][0] for i in range(len(df))]
    
    # Get fingerprints and load model
    finger_list = list(df["fp_3_4096"])
    model = Doc2Vec.load("../../model/fpdoc2vec4096.model")
    
    # Generate compound vectors
    compound_vec = addvec(finger_list, model)
    
    # Calculate similarities to sucrose
    target_compound = "sucrose"
    df[target_compound] = calculate_similarities(df, compound_vec, target_compound)
    
    # Get top 10 similar compounds
    top_similar = get_top_similar_compounds(df, target_compound, 10)
    
    # Display results in the same format as the previous code
    print(f"・Similar terms to '{target_compound}' with similarity scores:")
    for idx, row in top_similar.iterrows():
        compound_name = row['NAME']
        similarity_score = row[target_compound]
        print(f"  {compound_name}: {similarity_score:.4f}")

if __name__ == "__main__":
    main()
