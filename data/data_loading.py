import pickle
import time
import requests
from typing import Dict, List, Optional, Tuple, Union
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import PandasTools
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd


def smiles_to_cid(smiles: str, return_single: bool = True) -> Union[int, List[int], None]:
    """Convert SMILES to PubChem CID
    
    Args:
        smiles: String containing the SMILES notation of a molecule
        return_single: Boolean indicating whether to return a single CID (True) or a list of CIDs (False)
        
    Returns:
        A single CID (integer), a list of CIDs, or None if an error occurs
    """
    try:
        compounds = pcp.get_compounds(smiles, "smiles")
        cids = list(map(lambda x: x.cid, compounds))
    except ValueError as e:
        return None
    if return_single:
        return cids[0]
    else:
        return cids


def fetch_compound_descriptions(df: pd.DataFrame) -> Dict[Tuple[str, str, str], str]:
    """Fetch descriptions from PubChem using various identifiers
    
    Args:
        df: DataFrame containing compound information with columns 'inchikey', 'cid', 'smiles', and 'NAME'
        
    Returns:
        Dictionary with keys as tuples (NAME, smiles, inchikey) and values as compound descriptions
    """
    descriptions = {}
    temp_array1 = df['inchikey'].tolist()
    temp_array2 = df['cid'].tolist()
    temp_array3 = df["smiles"].tolist()
    temp_array4 = df['NAME'].tolist()
    
    for inchikey, cid, smiles, NAME in tqdm(zip(temp_array1, temp_array2, temp_array3, temp_array4)):
        url1 = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/" + str(inchikey) + "/description/XML"
        url2 = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/" + str(cid) + "/description/XML"
        url3 = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/" + str(smiles) + "/description/XML"
        url4 = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/" + str(NAME) + "/description/XML"
        urls = [url1, url2, url3, url4]
        responses = list(map(lambda x: requests.get(x), urls))
        soups = list(map(lambda x: BeautifulSoup(x.text, "xml"), responses))
        for soup in soups:
            try:
                descriptions[(NAME, smiles, inchikey)] = soup.find("Description").get_text()
                break
            except:
                continue
    return descriptions


def mol_to_inchikey(mol_list: List) -> List[Union[str, int]]:
    """Convert RDKit molecule objects to InChIKeys
    
    Args:
        mol_list: List of RDKit molecule objects
        
    Returns:
        List of InChIKeys as strings, with 0 for molecules that couldn't be converted
    """
    inchikeys = []
    for i in mol_list:
        try:
            inchikeys.append(Chem.MolToInchiKey(i))
        except:
            inchikeys.append(0)
    return inchikeys


def main(sdf_file: str, name_line: str, mol_file_line: str, file_name: str) -> None:
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


if __name__ == "__main__":
    # Example usage - replace with your actual file paths
    sdf_file = "chebi_file.sdf"
    file_name = "output_description.pkl"
    main(sdf_file, name_line = "ChEBI Name", mol_file_line = "ROMol", file_name) 
