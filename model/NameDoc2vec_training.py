import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from FpDoc2Vec_training.py import train_doc2vec_model

def lowercasing(x: Any) -> Any:
    """Convert input to lowercase, handling different data types
    
    Args:
        x: Input data which can be a string, list, tuple, or other convertible type
        
    Returns:
        The lowercase version of the input with the same structure
        
    Raises:
        Exception: If the input cannot be converted to lowercase
    """
    if isinstance(x, (list, tuple)):
        x = [lowercasing(_) for _ in x]
    elif isinstance(x, str):
        x = x.lower()
    else:
        try:
            x = str(x).lower()
        except Exception as e:
            raise Exception("Bugs") from e
    return x

def exact_name(df: pd.DataFrame) -> List[str]:
    """Extract and lowercase compound names from the dataframe
    
    Args:
        df: DataFrame containing a 'compounds' column where each entry is a list with compound name as first element
        
    Returns:
        List of lowercase compound names extracted from the dataframe
    """
    all_compounds = []
    for i in df["compounds"]:
        all_compounds.append(i[0])
    all_compounds = lowercasing(all_compounds)
    return all_compounds

def main(input_file: str, output_model_name: str, param: Dict[str, Any], purpose_description: str = "description_remove_stop_words") -> None:
    """Train a Doc2Vec model and save it
    
    Args:
        input_file: Path to a pickle file containing a pandas DataFrame with molecule data
        output_model_name: File path where the trained Doc2Vec model will be saved
        param: Dictionary of parameters for the Doc2Vec model initialization
        purpose_description: Column name in DataFrame containing preprocessed text for model training
                            (default: "description_remove_stop_words")
        
    Returns:
        None
    """
    # Load dataset
    with open(input_file, "rb") as f:
        df = pickle.load(f)
    
    # Extract compound names
    allcompounds = exact_name(df)
    
    # Define categories (not used in this code but kept for reference)
    categories = ['antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin', 
                 'flavouring_agent', 'agrochemical', 'volatile_oil', 'antibacterial_agent', 'insecticide']
    
    # Train Doc2Vec model with the specified text column
    model = train_doc2vec_model(df, allcompounds, param, purpose_description)
    
    # Save the model
    model.save(output_model_name)
    
    return None


if __name__ == "__main__":
    # Example usage - replace with your actual params
    param = {"vector_size": 100, 
         "min_count": 0,
         "window": 10,
         "min_alpha": 0.023491749982816976,
         "sample": 7.343338709169564e-06,
         "epochs": 859,
         "negative": 2,
         "ns_exponent": 0.8998927133390002,
         "workers": 1, 
         "seed": 100}
    
    # Example usage - replace with your actual file paths
    input_file = "10genre_dataset.pkl"
    output_model_name = "namedoc2vec.model"
    main(input_file, output_model_name, param)
