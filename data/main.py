from data.get_description import smiles_to_cid, fetch_compound_descriptions, mol_to_inchikey, main_data_loading
from data.preprocessing import lowercasing, split_sentence, split_word, cleanups, phrasing, phrase, main_preprocessing
from make_dataset import add_property_column, generate_morgan_fingerprints, make_dataset
from data.splitting import split_and_save_dataset

# Example usage - replace with your actual file paths
sdf_file = "chebi_file.sdf"
description_filename = "output_description.pkl"
main_data_loading(sdf_file, name_line="ChEBI Name", mol_file_line="ROMol", file_name=description_filename)

# Example usage - replace with your actual file paths
processed_filename = "processed_descriptions.pkl"
main_preprocessing(description_filename, processed_filename)

# Add property columns for multiple chemical roles
# Note: Replace these file names with your actual names after verification
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
dataset_file = "10genre_dataset.pkl"
make_dataset(processed_filename, properties, dataset_file)

# Split into training (90%) and test (10%) sets
train_file = "train_df.pkl"
test_file = "test_df.pkl"
split_and_save_dataset(dataset_file, test_file, train_file)
