import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from typing import Dict, List, Tuple, Any, Union, Optional
from gensim.models.doc2vec import Doc2Vec

def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from pickle file
    
    Args:
        file_path: Path to the pickle file containing the dataset
        
    Returns:
        DataFrame containing the loaded dataset
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def add_vectors(fp_list: List[List[int]], model: Doc2Vec) -> List[np.ndarray]:
    """Combine document vectors based on fingerprints
    
    Args:
        fp_list: List of fingerprint lists, where each fingerprint is represented as a list of indices
        model: Trained Doc2Vec model containing document vectors
        
    Returns:
        List of compound vectors as numpy arrays
    """
    compound_vec = []
    for i in fp_list:
        fingerprint_vec = 0
        for j in i:
            fingerprint_vec += model.dv.vectors[j]
        compound_vec.append(fingerprint_vec)
    return compound_vec

def generate_umap_embedding(vectors: List[np.ndarray], n_components: int = 2, 
                           n_neighbors: int = 50, min_dist: float = 1) -> np.ndarray:
    """Generate UMAP embedding from input vectors
    
    Args:
        vectors: List of vectors to embed
        n_components: Number of dimensions in the embedding
        n_neighbors: Number of neighbors to consider during embedding
        min_dist: Minimum distance between points in the embedding
        
    Returns:
        NumPy array containing the embedded coordinates
    """
    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors, 
        min_dist=min_dist,
        metric='cosine',
        random_state=0
    )
    return umap_model.fit_transform(vectors)


def plot_chemical_categories(df: pd.DataFrame, dim_df: pd.DataFrame, 
                            categories: List[str], categories_display: List[str], 
                            output_file: Optional[str] = None) -> None:
    """Create multi-panel plot for different chemical categories
    
    Args:
        df: DataFrame containing compound information and category labels
        dim_df: DataFrame containing the 2D embedding coordinates
        categories: List of category column names in df
        categories_display: List of category display names for plot titles
        output_file: Path where the output figure will be saved. If None, figure is not saved
        
    Returns:
        None
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, category in enumerate(categories):
        ax = axes[idx]
        
        # Create DataFrame with category labels and coordinates
        names_tb = pd.DataFrame(
            {"NAME": [i[0] for i in df["compounds"]], 
             "category": [1 if i == category else 0 for i in df[category]]}
        )
        index_tb = pd.concat([names_tb, dim_df], axis=1)
        
        # Plot non-category points (blue)
        mask_0 = index_tb["category"] == 0
        ax.scatter(index_tb[mask_0]["x"], index_tb[mask_0]["y"], 
                  c='blue', s=9, alpha=0.6, label='non')
        
        # Plot category points (red)
        mask_1 = index_tb["category"] == 1
        ax.scatter(index_tb[mask_1]["x"], index_tb[mask_1]["y"], 
                  c='red', s=9, alpha=1, label=category)
        
        ax.set_title(categories_display[idx], fontsize=21, fontweight='bold')
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.tick_params(axis='both', which='major', labelsize=8)
    
    plt.tight_layout()
    
    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return None


def main(input_path: str, fpdoc2vecmodel_path: str, output_path: str) -> None:
    """Process chemical data, generate UMAP embedding, and create visualization
    
    Args:
        input_path: Path to the input pickle file containing chemical dataset
        fpdoc2vecmodel_path: Path to the saved Doc2Vec model
        output_path: Path where the output visualization will be saved
        
    Returns:
        None
    """
    # Load the dataset
    df = load_data(input_path)
    
    # Extract fingerprints and load Doc2Vec model
    finger_list = list(df["fp_3_4096"])
    model = Doc2Vec.load(fpdoc2vecmodel_path)
    
    # Generate compound vectors
    compound_vec = add_vectors(finger_list, model)
    vec = np.array(compound_vec)
    
    # Generate UMAP embedding
    digits_tsne = generate_umap_embedding(vec)
    dim_df = pd.DataFrame(digits_tsne, columns=["x", "y"])
    
    # Define categories and their display names
    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin',
        'flavouring_agent', 'agrochemical', 'volatile_oil', 'antibacterial_agent', 'insecticide'
    ]
    categories_display = [
        '"antioxidant"', '"anti-inflammatory agent"', '"allergen"', '"dye"', '"toxin"',
        '"flavouring agent"', '"agrochemical"', '"volatile oil"', '"antibacterial agent"', '"insecticide"'
    ]
    
    # Create and save plot
    plot_chemical_categories(df, dim_df, categories, categories_display, output_path)


if __name__ == "__main__":
    input_path = "10genre_dataset.pkl"
    fpdoc2vecmodel_path = "fpdoc2vec.model"
    output_path = "fpdoc2vec_umap.png"
    main(input_path, fpdoc2vecmodel_path, output_path)
