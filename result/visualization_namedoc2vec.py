import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from gensim.models.doc2vec import Doc2Vec

def load_data(file_path):
    """Load dataset from pickle file"""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def load_doc2vec_vectors(model, num_vectors):
    """Extract document vectors from a trained Doc2Vec model"""
    indices = [i for i in range(num_vectors)]
    vectors = np.array([model.dv.vectors[i] for i in indices])
    return vectors

def generate_umap_embedding(vectors, n_components=2, n_neighbors=50, min_dist=1):
    """Generate UMAP embedding from input vectors"""
    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors, 
        min_dist=min_dist,
        metric='cosine',
        random_state=0
    )
    return umap_model.fit_transform(vectors)

def plot_chemical_categories(df, dim_df, categories, display_names, output_file=None):
    """Create multi-panel plot for different chemical categories"""
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
        ax.scatter(
            index_tb[mask_0]["x"], 
            index_tb[mask_0]["y"], 
            c='blue', 
            s=9, 
            alpha=0.6, 
            label='non'
        )
        
        # Plot category points (red)
        mask_1 = index_tb["category"] == 1
        ax.scatter(
            index_tb[mask_1]["x"], 
            index_tb[mask_1]["y"], 
            c='red', 
            s=9, 
            alpha=1, 
            label=category
        )
        
        # Configure plot aesthetics
        ax.set_title(display_names[idx], fontsize=21, fontweight='bold')
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.tick_params(axis='both', which='major', labelsize=8)
    
    plt.tight_layout()
    
    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    # Load dataset and model
    df = load_data("data/10genre_predict.pkl")
    model = Doc2Vec.load("model/namedoc2vec.model")
    
    # Extract vectors from Doc2Vec model
    vec = load_doc2vec_vectors(model, len(df))
    
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
    plot_chemical_categories(df, dim_df, categories, display_names, "namedoc2vec_umap.png")

if __name__ == "__main__":
    main()
