from result.SHAP.shapvalue_to_structure_core import visualize_shap_on_molecule
from utils import load_pickle

def main() -> None:
    """Load SHAP values and a compound dataset, then generate a SHAP-colored SVG structure image."""
    # Example usage with default values
    target_molecule_name = "quercetin" 
    shap_values_path = "antioxidant_fpdoc2vec.pkl"
    value = load_pickle(shap_values_path)
    data_path = "data/created_dataset/train_df.pkl"
    df = load_pickle(data_path)

    visualize_shap_on_molecule(
        compound_name = target_molecule_name, 
        df = df, 
        shap_values = value, 
        fp_radius = 3, 
        nBits = 4096, 
        compound_name_column = 'NAME',
        mol_column = 'ROMol',
        output = "quercetin_fp_doc2vec_shap.svg",
        size = (300, 300))

if __name__ == "__main__":
    main()