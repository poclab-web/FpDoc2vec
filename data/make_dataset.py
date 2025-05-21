import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Callable, Optional, Any, Tuple
from rdkit import Chem
from rdkit.Chem import PandasTools, AllChem
from data_loading import mol_to_inchikey
from preprocessing import lowercasing


def add_property_column(df: pd.DataFrame, property_name: str, sdf_path: str) -> pd.DataFrame:
    """Add a property column based on InChIKey matching with compounds from an SDF file
    
    Args:
        df: DataFrame containing compound information with an 'inchikey' column
        property_name: Name of the property to add (will be used as column name)
        sdf_path: Path to the SDF file containing compounds with the specified property
        
    Returns:
        DataFrame with the new property column added
    """
    property_df = PandasTools.LoadSDF(sdf_path)
    property_df["inchikey"] = mol_to_inchikey(property_df)
    df[property_name] = [property_name if i in list(property_df['inchikey']) else "No" for i in df["inchikey"]]
    return df


def generate_morgan_fingerprints(df: pd.DataFrame, radius: int, n_bits: int) -> List[List[int]]:
    """
    Generate Morgan fingerprints for molecules in the dataframe.
    
    Args:
        df: DataFrame containing RDKit molecule objects in a column named 'ROMol'
        radius: The radius of the Morgan fingerprint. Higher values capture more extended 
                connectivity information around each atom
        n_bits: The length of the bit vector. Larger values reduce the chance of bit collisions
    
    Returns:
        A list containing the indices of bits set to 1 for each molecule's fingerprint.
        Each inner list represents the active bits for a single molecule.
    """
    fingerprints = []
    for i, mol in enumerate(df["ROMol"]):
        try:
            fingerprint = [j for j in AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)]
            fingerprints.append(fingerprint)
        except:
            print(f"Error processing molecule at index {i}")
            continue
    fingerprints = np.array(fingerprints)
    return [[j for j in range(n_bits) if i[j] == 1] for i in fingerprints]

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
