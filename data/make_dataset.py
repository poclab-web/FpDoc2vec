import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

def mol2inchi(df):
    """Convert RDKit molecule objects to InChIKeys"""
    li = []
    for i in df["ROMol"]:
        try:
            li.append(Chem.MolToInchiKey(i))
        except:
            li.append(None)
    return li

def add_property_column(df, property_name, sdf_path):
    """Add a property column based on InChIKey matching with compounds from an SDF file"""
    property_df = PandasTools.LoadSDF(sdf_path)
    property_df["inchikey"] = mol2inchi(property_df)
    df[property_name] = [property_name if i in list(property_df['inchikey']) else "No" for i in df["inchikey"]]
    return df

def main():
    # Load previously processed data
    with open("3starAll_text_ver2.pkl", "rb") as f:
        df = pickle.load(f)
    
    # Consistency check and duplicate handling
    dup_df = df[df.duplicated(subset="description", keep=False)]
    dup_df["NAME"] = [lowercasing(i[0]).replace(" ", "_") for i in a["compounds"]]
    filtered_df = dup_df[dup_df.apply(lambda x: x['description_split'][0][0] == x['NAME'], axis=1)]
    supple_df = dup_df.loc[[3829, 40666, 11662, 8371, 4430, 25339]]
    comp_df = pd.concat([filtered_df, supple_df])
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
        "antioxidant": "data/ChEBI_antioxidant.sdf",
        "anti_inflammatory": "data/ChEBI_anti_inflammatory_agent.sdf",
        "allergen": "data/ChEBI_allergen.sdf",
        "dye": "data/ChEBI_dye.sdf",
        "toxin": "data/ChEBI_toxin.sdf",
        "flavouring_agent": "data/ChEBI_flavouring_agent.sdf",
        "agrochemical": "data/ChEBI_agrochemical.sdf",
        "volatile_oil": "data/ChEBI_volatile_oil_component.sdf",
        "antibacterial_agent": "data/ChEBI_antibacterial_agent.sdf",
        "insecticide": "data/ChEBI_insecticide.sdf"
    }
    
    for property_name, sdf_path in properties.items():
        df = add_property_column(df, property_name, sdf_path)
    
    # Filter compounds that have at least one property
    target_columns = [
        "antioxidant", "anti_inflammatory", "allergen", "dye", "toxin", 
        "flavouring_agent", "agrochemical", "volatile_oil", "antibacterial_agent", "insecticide"
    ]
    
    df = df[df[target_columns].ne("No").any(axis=1)].reset_index(drop=True)
    
    # Save processed dataset
    with open("10genre_dataset.pkl", "wb") as f:
        pickle.dump(df, f)  

if __name__ == "__main__":
    main()
