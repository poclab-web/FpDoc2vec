import pickle
import pandas as pd
from typing import List, Optional

def split_and_save_dataset(input_filename: str, testfile_name: str, trainfile_name: str) -> None:
    """
    Split the chemical compound dataset into training and test sets for novel molecule prediction evaluation.
    
    Args:
        input_filename: Path to the pickle file for DataFrame for prediction
        testfile_name: Path where the test dataset pickle file will be saved (10% of data)
        trainfile_name: Path where the training dataset pickle file will be saved (90% of data)
        
    Returns:
        None
    """
    # Load the original dataset
    with open(input_filename, "rb") as f:
        df = pickle.load(f)
    
    # Define categories of interest
    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin', 
        'flavouring_agent', 'agrochemical', 'volatile_oil', 
        'antibacterial_agent', 'insecticide'
    ]
    
    # Split dataset: 10% for testing, 90% for training
    df1 = df.sample(frac=0.1)  # Test set
    df2 = df.drop(df1.index)   # Training set
    
    # Reset indices
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    
    # Save test and training datasets
    with open(testfile_name, 'wb') as f:
        pickle.dump(df1, f)
    
    with open(trainfile_name, 'wb') as f:
        pickle.dump(df2, f)

if __name__ == "__main__":
    # Example usage - replace with your actual file paths
    input_filename = "10genre_dataset.pkl"
    testfile_name = "test_df.pkl"
    trainfile_name = "train_df.pkl"
    split_and_save_dataset(input_filename, testfile_name, trainfile_name)
