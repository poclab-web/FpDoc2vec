import pickle
import time
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from rdkit import Chem
from rdkit.Chem import PandasTools
import pubchempy as pcp
import pandas as pd

def smiles_to_cid(smiles, return_single=True):
    """Convert SMILES to PubChem CID"""
    try:
        compounds = pcp.get_compounds(smiles, "smiles")
        cids = list(map(lambda x: x.cid, compounds))
    except ValueError as e:
        return None
    if return_single:
        return cids[0]
    else:
        return cids
        
def fetch_compound_descriptions(df):
    """Fetch descriptions from PubChem using various identifiers"""
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

def mol_to_inchikey(mol_list):
    """Convert RDKit molecule objects to InChIKeys"""
    inchikeys = []
    for i in mol_list:
        try:
            inchikeys.append(Chem.MolToInchiKey(i))
        except:
            inchikeys.append(0)
    return inchikeys

def main(input_sdf_file, output_file_name):
    # Load chemical data
    df = PandasTools.LoadSDF(sdf_file)
    data_df = df[["ChEBI Name", "ROMol"]].rename(columns={"ChEBI Name":"NAME"})
    
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
    main(input_sdf_file, output_file_name)
