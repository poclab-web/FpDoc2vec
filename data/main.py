import pickle
import time
import re
from typing import Dict, List, Tuple, Union, Any, Optional, Callable
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import PandasTools, AllChem
from gensim.parsing.preprocessing import remove_stopword_tokens
from gensim.models.phrases import Phrases

from data_loading import smiles_to_cid, fetch_compound_descriptions, mol_to_inchikey
from data_preprocessing import lowercasing, split_sentence, split_word, cleanups, phrasing, phrase
from make_dataset import add_property_column, generate_morgan_fingerprints


def data_loding(sdf_file: str, name_line: str, mol_file_line: str, file_name: str) -> None:
    """Process SDF file to extract compound information and fetch descriptions from PubChem
    
    Args:
        sdf_file: Path to SDF file containing compound data obtained from ChEBI
        name_line: A column name in a dataframe that stores compound names
        mol_file_line: A column name in a dataframe that stores molfile
        file_name: Path where the output pickle file will be saved
        
    Returns:
        None
    """
    # Load chemical data
    df = PandasTools.LoadSDF(sdf_file)
    data_df = pd.DataFrame({"NAME": list(df[name_line]), "ROMol": list(df[mol_file_line])})
    
    # Generate InChIKeys
    data_df["inchikey"] = mol_to_inchikey(data_df["ROMol"])
    
    # Remove entries with invalid InChIKey
    data_df = data_df.dropna(subset=['inchikey'])
    
    # Remove zwitterion entries
    data_df = data_df[~data_df['NAME'].str.contains('zwitterion')]
    
    # Remove duplicate InChIKeys
    data_df = data_df.drop_duplicates(subset="inchikey", keep="first")
    
    # Generate SMILES and fetch CIDs
    data_df["smiles"] = data_df["ROMol"].map(Chem.MolToSmiles)
    
    # Get PubChem CIDs
    cids = []
    for i in tqdm(data_df["smiles"]):
        try:
            cids.append(smiles_to_cid(i))
        except:
            cids.append(None)
    data_df["cid"] = cids
    
    # Process data in batches
    data_df1 = data_df[:25000]
    data_df2 = data_df[25000:]
    
    # Fetch descriptions for first batch
    all1 = fetch_compound_descriptions(data_df1)
    
    # Wait to avoid API rate limiting
    time.sleep(3600)
    
    # Fetch descriptions for second batch
    all2 = fetch_compound_descriptions(data_df2)
    
    # Combine results
    all1.update(all2)
    
    # Save data
    with open(file_name, "wb") as f:
        pickle.dump(all1, f)

def preprocessing(input_file: str, output_filename: str) -> pd.DataFrame:
    """Load and preprocess chemical compound descriptions
    
    Args:
        input_file: Path to the pickle file containing compound descriptions
        output_filename: Path where the processed data will be saved
        
    Returns:
        DataFrame containing the processed chemical descriptions
    """
    # Load the data
    with open(input_file, "rb") as f:
        dict_data = pickle.load(f)
        all_text_df = pd.DataFrame(dict_data.items(), columns=["compounds", "description"])
    
    # Apply preprocessing steps
    all_text_df["description_lower"] = all_text_df["description"].map(lambda x: lowercasing(x))
    all_text_df["description_split_sentence"] = all_text_df["description_lower"].map(lambda x: split_sentence(x))
    all_text_df["description_split"] = all_text_df["description_split_sentence"].map(lambda x: split_word(x))
    all_text_df["description_remove_stop_words"] = all_text_df["description_split"].map(lambda x: cleanups(x))
    
    # Apply phrasing with compound names
    li = []
    for i in tqdm(range(len(all_text_df))):
        li.append(phrasing(all_text_df.iat[i, 5], phrase_list=[all_text_df.iat[i, 0][0]]))
    all_text_df["description_phrases"] = li
    all_text_df["description_phrases"] = all_text_df["description_phrases"].map(lambda x: phrase(x, 1, 0.7))

    # Apply gensim phrase detection
    all_text_df["description_gensim"] = all_text_df["description_remove_stop_words"].map(lambda x: phrase(x, 1, 0.7))
    
    # Save the processed data
    with open(output_filename, "wb") as f:
        pickle.dump(all_text_df, f)
    
    return all_text_df
  
def make_dataset(input_file: str, properties: Dict[str, str], output_file: str) -> None:
    """Process chemical data, add property columns, and generate fingerprints
    
    Args:
        input_file: Path to the input pickle file containing preprocessed chemical data
        properties: Dictionary mapping property names to SDF file paths
        output_file: Path where the output pickle file will be saved
        
    Returns:
        None
    """
    # Load previously processed data
    with open(input_file, "rb") as f:
        df = pickle.load(f)
    
    # Extract records with duplicate descriptions
    dup_df = df[df.duplicated(subset="description", keep=False)]
    # Convert compound names to lowercase and replace spaces with underscores
    dup_df["NAME"] = [lowercasing(i[0]).replace(" ", "_") for i in a["compounds"]]
    # Filter records where the first word of the description matches the compound name
    filtered_df = dup_df[dup_df.apply(lambda x: x['description_split'][0][0] == x['NAME'], axis=1)]
    # Add manually verified records that weren't captured by the automatic filtering
    # Note: Replace these indices with your actual indices after verification
    supple_df = dup_df.loc[[3829, 40666, 11662, 8371, 4430, 25339]]
    # Combine automatically filtered and manually verified records
    comp_df = pd.concat([filtered_df, supple_df])
    # Identify duplicate records to be removed (those not in the combined set)
    del_df = dup_df[~dup_df["inchikey"].isin(list(comp_df["inchikey"]))]
    
    # Normalization
    df = df[~df["inchikey"].isin(list(del_df["inchikey"]))]
    df = df.reset_index(drop=True)
    
    # Extract SMILES and InChIKey data
    df["smiles"] = [i[1] for i in df["compounds"]]
    PandasTools.AddMoleculeColumnToFrame(df, "smiles")
    df.insert(1, "inchikey", [i[2] for i in df["compounds"]])
    
    for property_name, sdf_path in properties.items():
        df = add_property_column(df, property_name, sdf_path)
    
    # Filter compounds that have at least one property
    target_columns = [
        "antioxidant", "anti_inflammatory", "allergen", "dye", "toxin", 
        "flavouring_agent", "agrochemical", "volatile_oil", "antibacterial_agent", "insecticide"
    ]
    
    df = df[df[target_columns].ne("No").any(axis=1)].reset_index(drop=True)
    
    # Generate fingerprints
    df["fp_3_4096"] = generate_morgan_fingerprints(df, 3, 4096)
    finger_list = list(df["fp_3_4096"])
    
    # Save processed dataset
    with open(output_file, "wb") as f:
        pickle.dump(df, f)

if __name__ == "__main__":
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
