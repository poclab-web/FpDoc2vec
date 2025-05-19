import pickle
import numpy as np
from rdkit.Chem import AllChem
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def generate_morgan_fingerprints(df: pd.DataFrame, radius: int, n_bits: int) -> List[List[int]]:
    """
    Generate Morgan fingerprints (ECFP6) for molecules in the dataframe.
    
    Args:
        df: DataFrame containing RDKit molecule objects in a column named 'ROMol'
        radius: The radius of the Morgan fingerprint. Higher values capture more extended 
                connectivity information around each atom
        n_bits: The length of the bit vector. Larger values reduce the chance of bit collisions
    
    Returns:
        A list containing the indices of bits set to 1 for each molecule's fingerprint.
        Each inner list represents the active bits for a single molecule.
    """
    fingerprints = []
    for i, mol in enumerate(df["ROMol"]):
        try:
            fingerprint = [j for j in AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)]
            fingerprints.append(fingerprint)
        except:
            print(f"Error processing molecule at index {i}")
            continue
    fingerprints = np.array(fingerprints)
    return [[j for j in range(n_bits) if i[j] == 1] for i in fingerprints]

def train_fingerprint_doc2vec_model(df: pd.DataFrame, fingerprints: List[List[int]], param: Dict[str, Any]) -> Doc2Vec:
    """
    Train a Doc2Vec model using document descriptions and molecular fingerprints as tags.
    
    Args:
        df: DataFrame containing preprocessed descriptions in a column named 'description_remove_stop_words'
        fingerprints: List of lists where each inner list contains the active bit indices for a molecule's fingerprint
        param: Dictionary of parameters for the Doc2Vec model initialization
        
    Returns:
        A trained Doc2Vec model
    """
    corpus = [sum(doc, []) for doc in df["description_remove_stop_words"]]
    tagged_documents = [TaggedDocument(words=corpus, tags=fingerprints[i]) for i, corpus in enumerate(corpus)]
    
    model = Doc2Vec(tagged_documents, **param)
    
    return model

def main(input_file: str, output_model_name: str, param: Dict[str, Any]) -> None:
    """
    Load data, generate molecular fingerprints, train a Doc2Vec model and save it.
    
    Args:
        input_file: Path to a pickle file containing a pandas DataFrame with molecule data
        output_model_name: File path where the trained Doc2Vec model will be saved
        param: Dictionary of parameters for the Doc2Vec model initialization
        
    Returns:
        None
    """
    # Load dataset
    with open(input_file, "rb") as f:
        df = pickle.load(f)
    
    # Generate fingerprints
    df["fp_3_4096"] = generate_morgan_fingerprints(df, 3, 4096)
    finger_list = list(df["fp_3_4096"])
    
    # Define categories (not used in this code but kept for reference)
    categories = ['antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin', 
                 'flavouring_agent', 'agrochemical', 'volatile_oil', 'antibacterial_agent', 'insecticide']
    
    # Train Doc2Vec model
    model = train_fingerprint_doc2vec_model(df, finger_list, param)
    
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
    output_model_name = "fpdoc2vec.model"
    main(input_file, output_model_name)
