import pickle
from typing import Tuple
import pandas as pd
import numpy as np


def load_descriptors(
    descriptor_path: str,
    train_df_path: str,
    test_df_path: str):
    
    with open(descriptor_path, "rb") as f:
        df = pickle.load(f)

    with open(train_df_path, "rb") as f:
        train_df = pickle.load(f)
    with open(test_df_path, "rb") as f:
        test_df = pickle.load(f)

    train_desc_df = df[df["inchikey"].isin(list(train_df["inchikey"]))]
    test_desc_df = df[df["inchikey"].isin(list(test_df["inchikey"]))]

    return train_desc_df, test_desc_df


def main():
    train_desc_df, test_desc_df = load_descriptors(
        descriptor_path="descriptors.pkl",
        train_df_path="train_df.pkl",
        test_df_path="test_df.pkl"
    )
    with open("data/Descriptor/train_desc_df.pkl", "wb") as f:
        pickle.dump(train_desc_df, f)
    with open("data/Descriptor/test_desc_df.pkl", "wb") as f:
        pickle.dump(test_desc_df, f)

if __name__ == "__main__":
    main()