from data_loading import smiles_to_cid, fetch_compound_descriptions, mol_to_inchikey, main_data_loding
from data_preprocessing import lowercasing, split_sentence, split_word, cleanups, phrasing, phrase
from make_dataset import add_property_column, generate_morgan_fingerprints

# Example usage - replace with your actual file paths
sdf_file = "chebi_file.sdf"
description_filename = "output_description.pkl"
main_get_description(sdf_file, name_line = "ChEBI Name", mol_file_line = "ROMol", description_filename) 

# Example usage - replace with your actual file paths
processed_filename = "processed_descriptions.pkl"
preprocess_chemical_descriptions(description_filename, processed_filename)

# Add property columns for multiple chemical roles
# Note: Replace these file name with your actual name after verification
properties = {
    "antioxidant": "chemdata/ChEBI_antioxidant.sdf",
    "anti_inflammatory": "chemdata/ChEBI_anti_inflammatory_agent.sdf",
    "allergen": "chemdata/ChEBI_allergen.sdf",
    "dye": "chemdata/ChEBI_dye.sdf",
    "toxin": "chemdata/ChEBI_toxin.sdf",
    "flavouring_agent": "chemdata/ChEBI_flavouring_agent.sdf",
    "agrochemical": "chemdata/ChEBI_agrochemical.sdf",
    "volatile_oil": "chemdata/ChEBI_volatile_oil_component.sdf",
    "antibacterial_agent": "chemdata/ChEBI_antibacterial_agent.sdf",
    "insecticide": "chemdata/ChEBI_insecticide.sdf"
}
# Example usage - replace with your actual file paths
output_file = "10genre_dataset.pkl"  
make_dataset(processed_filename, properties, output_file)
