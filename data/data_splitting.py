import pickle
import pandas as pd

def split_and_save_dataset():
    """
    Split the chemical compound dataset into training and test sets for novel molecule prediction evaluation.
    """
    # Load the original dataset
    with open("chemdata/10genre_dataset.pkl", "rb") as f:
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
    with open("test_df.pkl", 'wb') as f:
        pickle.dump(df1, f)
    
    with open("train_df.pkl", 'wb') as f:
        pickle.dump(df2, f)

if __name__ == "__main__":
    split_and_save_dataset()
