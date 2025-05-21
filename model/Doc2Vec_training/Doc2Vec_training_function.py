import pickle
import numpy as np
from rdkit.Chem import AllChem
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def generate_morgan_fingerprints(df: pd.DataFrame, radius: int, n_bits: int) -> List[List[int]]:
    """
    Generate Morgan fingerprints for molecules in the dataframe.
    
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

def train_doc2vec_model(df: pd.DataFrame, tag_list: List[List[int]], param: Dict[str, Any], purpose_description: str) -> Doc2Vec:
    """
    Train a Doc2Vec model using document descriptions as tags.
    
    Args:
        df: DataFrame containing preprocessed descriptions 
        tag_list: List of lists where each inner list contains the tags for a document
        param: Dictionary of parameters for the Doc2Vec model initialization
        purpose_description: Column name in df containing the preprocessed text to use for training
        
    Returns:
        A trained Doc2Vec model
    """
    corpus = [sum(doc, []) for doc in df[purpose_description]]
    tagged_documents = [TaggedDocument(words=corpus, tags=[tag_list[i]]) for i, doc in enumerate(corpus)]
    
    model = Doc2Vec(tagged_documents, **param)
    
    return model

def main_doc2vec(input_file: str, output_model_name: str, param: Dict[str, Any], tag_list: List[str], purpose_description: str = "description_remove_stop_words") -> None:
    """
    Load data, generate molecular fingerprints, train a Doc2Vec model and save it.
    
    Args:
        input_file: Path to a pickle file containing a pandas DataFrame with molecule data
        output_model_name: File path where the trained Doc2Vec model will be saved
        param: Dictionary of parameters for the Doc2Vec model initialization
        tag_list: List of lists where each inner list contains the tags for a document
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
    model = train_doc2vec_model(df, tag_list, param, purpose_description)
    
    # Save the model
    model.save(output_model_name)
    
    return None
