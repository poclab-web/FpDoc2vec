import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from utils import CATEGORIES


def generate_umap_embedding(vectors: List[np.ndarray], n_neighbors: int, min_dist: float, n_components: int = 2) -> np.ndarray:
    """Generate UMAP embedding from input vectors"""
    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors, 
        min_dist=min_dist,
        metric='cosine',
        random_state=0
    )
    return umap_model.fit_transform(vectors)


def plot_chemical_categories(df: pd.DataFrame, dim_df: pd.DataFrame, output_file: str) -> None:
    """Create multi-panel plot for different chemical categories"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, category in enumerate(CATEGORIES):
        ax = axes[idx]
        
        # Create DataFrame with category labels and coordinates
        names_tb = pd.DataFrame(
            { "NAME": [i[0] for i in df["compounds"]], "category": (df[category] == category).astype(int)}
        )
        index_tb = pd.concat([names_tb, dim_df], axis=1)
        
        # Plot non-category points (blue)
        mask_0 = index_tb["category"] == 0
        ax.scatter(index_tb[mask_0]["x"], index_tb[mask_0]["y"], c='blue', s=9, alpha=0.6, label='non')
        
        # Plot category points (red)
        mask_1 = index_tb["category"] == 1
        ax.scatter(index_tb[mask_1]["x"], index_tb[mask_1]["y"], c='red', s=9, alpha=1, label=category)
        
        ax.set_title(category, fontsize=21, fontweight='bold')
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.tick_params(axis='both', which='major', labelsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


def run_visualization(df: pd.DataFrame, vec: Dict[str, np.ndarray], output_path: str, n_neighbors: int, min_dist: float) -> None:
    """Process chemical data and generate UMAP visualization of chemical categories"""
    
    # Generate UMAP embedding
    umap_result = generate_umap_embedding(vec, n_neighbors, min_dist)
    dim_df = pd.DataFrame(umap_result, columns=["x", "y"])
    
    # Create visualization
    plot_chemical_categories(df, dim_df, output_path)
