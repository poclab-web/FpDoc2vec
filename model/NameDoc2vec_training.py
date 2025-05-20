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

def exact_name(df):
    """Extract and lowercase compound names from the dataframe"""
    all_compounds = []
    for i in df["compounds"]:
        all_compounds.append(i[0])
    all_compounds = lowercasing(all_compounds)
    return all_compounds

def main():
    # Load data
    with open(input_file, "rb") as f:
        df = pickle.load(f)
    allcompounds = exact_name(df)
    
    # Define chemical categories
    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin',
        'flavouring_agent', 'agrochemical', 'volatile_oil', 'antibacterial_agent', 'insecticide'
    ]
    
    # Train and save the Doc2Vec model
    model = train_doc2vec_model(df, allcompounds, )
    # Save the model
    model.save("namedoc2vec.model")

if __name__ == "__main__":
    main()
