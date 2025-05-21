from shapvalue_to_structure import visualize_shap_on_molecule, main

# Example usage with default values
shap_values_path = "shap_value_ecfp.pkl"
chemical_data_path = "10genre_dataset.pkl"
target_molecule = "quercetin" # Please modify according to the purpose.

main(shap_values_path, chemical_data_path=chemical_data_path, target_molecule, fp_radius=3, nBits=4096, scale_factor=1.0)

# Example usage with default values
shap_values_path = "shap_value_doc2vec.pkl"
chemical_data_path = "10genre_dataset.pkl"
target_molecule = "quercetin" # Please modify according to the purpose.
main(shap_values_path, chemical_data_path, target_molecule, fp_radius=3, nBits=4096, scale_factor=1.0)
