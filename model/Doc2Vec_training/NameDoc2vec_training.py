import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from FpDoc2Vec_training import train_doc2vec_model

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
