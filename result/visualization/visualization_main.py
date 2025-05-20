import pandas as pd
import matplotlib.pyplot as plt
import umap
from typing import Dict, List, Tuple, Any, Union, Optional, Mapping
from gensim.models.doc2vec import Doc2Vec

from visualization_function import load_data, add_vectors, generate_umap_embedding, plot_chemical_categories, make_name2vector, make_fp2vector


def main(input_path: str, vec: Dict[str, np.ndarray], output_path: str) -> None:
    """Process chemical data and generate UMAP visualization of chemical categories
    
    Args:
        input_path: Path to the input pickle file containing chemical dataset
        vec: Dictionary mapping compound IDs to their vector representations
        output_path: Path where the output visualization will be saved
        
    Returns:
        None
    """
    # Load dataset and model
    df = load_data(input_path)
    
    # Generate UMAP embedding
    umap_result = generate_umap_embedding(vec)
    dim_df = pd.DataFrame(umap_result, columns=["x", "y"])
    
    # Define categories and their display names
    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin',
        'flavouring_agent', 'agrochemical', 'volatile_oil', 'antibacterial_agent', 
        'insecticide'
    ]
    
    display_names = [
        '"antioxidant"', '"anti-inflammatory agent"', '"allergen"', '"dye"', 
        '"toxin"', '"flavouring agent"', '"agrochemical"', '"volatile oil"', 
        '"antibacterial agent"', '"insecticide"'
    ]
    
    # Create visualization
    plot_chemical_categories(df, dim_df, categories, display_names, output_path)
    
    return None


if __name__ == "__main__":
    
    #FpDoc2vec visualization
    input_path: str = "10genre_dataset.pkl"
    model_path: str = "fpdoc2vec.model"
    output_path: str = "fpdoc2vec_umap.png"
    df: pd.DataFrame = load_data(input_path)
    vec: Dict[str, np.ndarray] = make_fp2vector(model_path, df)
    main(input_path, vec, output_path)
    
    #NameDoc2vec visualization
    input_path: str = "10genre_dataset.pkl"
    model_path: str = "namedoc2vec.model"
    output_path: str = "namedoc2vec_umap.png"
    df: pd.DataFrame = load_data(input_path)
    vec: Dict[str, np.ndarray] = make_name2vector(model_path, df)
    main(input_path, vec, output_path)
