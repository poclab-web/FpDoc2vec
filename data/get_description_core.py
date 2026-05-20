import time
import requests
from typing import Dict, Tuple
from rdkit.Chem import PandasTools
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd


def load_sdf(sdf_file: str) -> pd.DataFrame:
    """Load compounds from SDF file and return cleaned DataFrame."""
    df = PandasTools.LoadSDF(sdf_file)
    data_df = pd.DataFrame({
        "NAME": list(df["ChEBI NAME"]),
        "inchikey": list(df["INCHIKEY"]),
        "ROMol": list(df["ROMol"]),
        "smiles": list(df["SMILES"]),
        "cid": [
            next((c.strip() for c in str(x).split(";") if c.strip()), None)
            for x in df["PubChem Compound Database Links"]
        ]
    })
    data_df = data_df.dropna(subset=['inchikey'])
    data_df = data_df[~data_df['NAME'].str.contains('zwitterion')]
    data_df = data_df.drop_duplicates(subset="inchikey", keep="first")
    return data_df


def _fetch_compound_descriptions(df: pd.DataFrame) -> Dict[Tuple[str, str, str], str]:
    descriptions = {}
    for inchikey, cid, smiles, NAME in tqdm(zip(df['inchikey'], df['cid'], df['smiles'], df['NAME'])):
        urls = [
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/description/XML",
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/description/XML",
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/description/XML",
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{NAME}/description/XML",
        ]
        for url in urls:
            soup = BeautifulSoup(requests.get(url).text, "xml")
            try:
                descriptions[(NAME, smiles, inchikey)] = soup.find("Description").get_text()
                break
            except:
                continue
    return descriptions


def fetch_descriptions(df: pd.DataFrame, batch_size: int = 25000) -> pd.DataFrame:
    """Fetch descriptions from PubChem and add as column to DataFrame."""
    all_descriptions = {}
    batches = [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]
    for idx, batch in enumerate(batches):
        all_descriptions.update(_fetch_compound_descriptions(batch))
        if idx < len(batches) - 1:
            time.sleep(3600)
    df["description"] = df.apply(
        lambda row: all_descriptions.get((row["NAME"], row["smiles"], row["inchikey"])), axis=1
    )
    return df