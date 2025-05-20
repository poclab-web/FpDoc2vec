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
