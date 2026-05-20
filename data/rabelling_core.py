import pandas as pd
from typing import Dict
from rdkit.Chem import PandasTools
from rdkit.Chem import rdMolDescriptors
from data.preprocessing_core import lowercasing


def add_property_column(df: pd.DataFrame, property_name: str, sdf_path: str) -> pd.DataFrame:
    """Add a binary property column based on InChIKey matching with an SDF file."""
    property_df = PandasTools.LoadSDF(sdf_path)
    property_df["inchikey"] = [rdMolDescriptors.CalcInchiKey(mol) if mol else None for mol in property_df["ROMol"]]
    df[property_name] = [property_name if i in list(property_df['inchikey']) else "No" for i in df["inchikey"]]
    return df


def main_rabelling(df: pd.DataFrame, properties: Dict[str, str]) -> pd.DataFrame:
    """Add property columns and filter to compounds with at least one property."""
    dup_df = df[df.duplicated(subset="description", keep=False)].copy()
    dup_df["name"] = [lowercasing(i).replace(" ", "_") for i in dup_df["NAME"]]
    filtered_df = dup_df[dup_df.apply(lambda x: x['description_split'][0][0] == x['name'], axis=1)]
    # Manually verified records — replace indices after verification
    supple_df = dup_df.loc[[3829, 40666, 11662, 8371, 4430, 25339]]
    comp_df = pd.concat([filtered_df, supple_df])
    del_df = dup_df[~dup_df["inchikey"].isin(list(comp_df["inchikey"]))]

    df = df[~df["inchikey"].isin(list(del_df["inchikey"]))].reset_index(drop=True)

    for property_name, sdf_path in properties.items():
        df = add_property_column(df, property_name, sdf_path)

    target_columns = list(properties.keys())
    df = df[df[target_columns].ne("No").any(axis=1)].reset_index(drop=True)

    return df
