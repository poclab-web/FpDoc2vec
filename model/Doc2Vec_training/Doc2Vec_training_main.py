import pickle
import numpy as np
from rdkit.Chem import AllChem
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from Doc2Vec_training_function import generate_morgan_fingerprints, lowercasing, exact_name, train_doc2vec_model

def main_doc2vec(input_file: str, output_model_name: str, param: Dict[str, Any], column_list: List[str], purpose_description: str = "description_remove_stop_words") -> None:
    """
    Load data, generate molecular fingerprints, train a Doc2Vec model and save it.
    
    Args:
        input_file: Path to a pickle file containing a pandas DataFrame with molecule data
        output_model_name: File path where the trained Doc2Vec model will be saved
        param: Dictionary of parameters for the Doc2Vec model initialization
        list: List of lists where each inner list contains the tags for a document
        purpose_description: Column name in DataFrame containing preprocessed text for model training (default: "description_remove_stop_words")
        
    Returns:
        None
    """
    # Load dataset
    with open(input_file, "rb") as f:
        df = pickle.load(f)
        
    # Define categories (not used in this code but kept for reference)
    categories = ['antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin', 
                 'flavouring_agent', 'agrochemical', 'volatile_oil', 'antibacterial_agent', 'insecticide']
    
    # Train Doc2Vec model with the specified text column
    model = train_doc2vec_model(df, list, param, purpose_description)
    
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


if __name__ == "__main__":
    # Example usage - replace with your actual params
    # Please change the parameter values as you like.
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
  
    # Building FpDoc2Vec model
    # Generate fingerprints
    df["fp_3_4096"] = generate_morgan_fingerprints(df, 3, 4096)
    finger_list = list(df["fp_3_4096"])
  
    # Example usage - replace with your actual file paths
    input_file = "10genre_dataset.pkl"
    output_model_name = "fpdoc2vec.model"
    main_doc2vec(input_file, output_model_name, param, finger_list)
  
    # Building NameDoc2Vec model
    # Extract compound names
    allcompounds = exact_name(df)
  
    # Example usage - replace with your actual file paths
    output_model_name = "namedoc2vec.model"
    main_doc2vec(input_file, output_model_name, param, allcompounds)
