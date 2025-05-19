import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from data_loading import mol_to_inchikey

def add_property_column(df, property_name, sdf_path):
    """Add a property column based on InChIKey matching with compounds from an SDF file"""
    property_df = PandasTools.LoadSDF(sdf_path)
    property_df["inchikey"] = mol_to_inchikey(property_df)
    df[property_name] = [property_name if i in list(property_df['inchikey']) else "No" for i in df["inchikey"]]
    return df

def generate_morgan_fingerprints(df):
    """
    Generate Morgan fingerprints (ECFP6) with radius 3 and 4096 bits for molecules in the dataframe.
    Returns a list of the bit positions that are set to 1 for each molecule.
    """
    fingerprints = []
    for i, mol in enumerate(df["ROMol"]):
        try:
            fingerprint = [j for j in AllChem.GetMorganFingerprintAsBitVect(mol, 3, 4096)]
            fingerprints.append(fingerprint)
        except:
            print("Error", i)
            continue
    fingerprints = np.array(fingerprints)
    return [[j for j in range(4096) if i[j] == 1] for i in fingerprints]

def main(input_file, output_file):
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
    
    # Add property columns for multiple chemical roles
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
    
    for property_name, sdf_path in properties.items():
        df = add_property_column(df, property_name, sdf_path)
    
    # Filter compounds that have at least one property
    target_columns = [
        "antioxidant", "anti_inflammatory", "allergen", "dye", "toxin", 
        "flavouring_agent", "agrochemical", "volatile_oil", "antibacterial_agent", "insecticide"
    ]
    
    df = df[df[target_columns].ne("No").any(axis=1)].reset_index(drop=True)
    
    # Generate fingerprints
    df["fp_3_4096"] = generate_morgan_fingerprints(df)
    finger_list = list(df["fp_3_4096"])
    
    # Save processed dataset
    with open(output_file, "wb") as f:
        pickle.dump(df, f)  

if __name__ == "__main__":
    input_file = "processed_descriptions.pkl"
    output_file = "10genre_dataset,pkl"
    main(input_file, output_file)
