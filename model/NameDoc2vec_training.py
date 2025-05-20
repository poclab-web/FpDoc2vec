import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def grouping(df):
    """Extract and lowercase compound names from the dataframe"""
    all_compounds = []
    for i in df["compounds"]:
        all_compounds.append(i[0])
    all_compounds = lowercasing(all_compounds)
    return all_compounds

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

def train_doc2vec_model(df):
    """Train a Doc2Vec model using compound descriptions"""
    # Extract compound names
    allcompounds = grouping(df)
    
    # Prepare corpus for Doc2Vec
    corpus = [sum(doc, []) for doc in df["description_remove_stop_words"]]
    
    # Create tagged documents
    tagged_documents = [
        TaggedDocument(words=doc, tags=[allcompounds[i]])
        for i, doc in enumerate(corpus)
    ]
    
    # Train Doc2Vec model with optimized hyperparameters
    model = Doc2Vec(
        tagged_documents,
        vector_size=100,
        min_count=0,
        window=10,
        min_alpha=0.023491749982816976,
        sample=7.343338709169564e-06,
        epochs=859,
        negative=2,
        ns_exponent=0.8998927133390002,
        workers=1,
        seed=100
    )
    
    # Save the trained model
    model.save("namedoc2vec.model")
    
    return model

def main():
    # Load data
    with open("data/10genre_dataset.pkl", "rb") as f:
        df = pickle.load(f)
    
    # Define chemical categories
    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin',
        'flavouring_agent', 'agrochemical', 'volatile_oil', 'antibacterial_agent', 'insecticide'
    ]
    
    # Train and save the Doc2Vec model
    model = train_doc2vec_model(df)
    # Save the model
    model.save("namedoc2vec.model")

if __name__ == "__main__":
    main()
