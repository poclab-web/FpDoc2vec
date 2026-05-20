import pandas as pd
from typing import Tuple


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into 90% train and 10% test sets."""
    test_df = df.sample(frac=0.1, random_state=0).reset_index(drop=True)
    train_df = df.drop(test_df.index).reset_index(drop=True)
    return train_df, test_df


def split_descriptors_dataset(
    descriptor_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load descriptor DataFrame and filter rows by train/test InChIKeys."""

    train_desc_df = descriptor_df[descriptor_df["inchikey"].isin(list(train_df["inchikey"]))]
    test_desc_df = descriptor_df[descriptor_df["inchikey"].isin(list(test_df["inchikey"]))]

    return train_desc_df, test_desc_df
