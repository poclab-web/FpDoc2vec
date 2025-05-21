from typing import List, Tuple, Any
from gensim.models.doc2vec import Doc2Vec


def find_similar_terms(model: Doc2Vec, query_term: str, top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Find terms that are most similar to the query term in the Doc2Vec model.
    
    Args:
        model: Trained Doc2Vec model
        query_term: Term to find similarities for
        top_n: Number of similar terms to return (default: 10)
        
    Returns:
        List of tuples containing (similar_term, similarity_score)
    """
    print(f"・Similar terms to '{query_term}' with similarity scores:")
    return model.dv.most_similar(positive=query_term, topn=top_n)

def similarity_namedoc2vec(input_path: str, model_path: str, target_compound: str = "sucrose", n: int = 10) -> None:
    """
    Load data and model, then demonstrate similarity search.
    
    Args:
        input_path: Path to the pickle file containing the dataset
        model_path: Path to the trained NameDoc2Vec model file
        target_compound: Name of the target compound (default: "sucrose")
        n: Number of similar compounds to return (default: 10)
        
    Returns:
        None
    """
    # Load dataset
    with open(input_path, "rb") as f:
        df = pickle.load(f)
    
    # Load pre-trained Doc2Vec model
    model = Doc2Vec.load(model_path)
    
    # Find terms similar to target compound
    similar_terms = find_similar_terms(model, target_compound, n)
    
    # Display results
    print(f"・Similar terms to '{target_compound}' with similarity scores:")
    for term, score in similar_terms:
        print(f"  {term}: {score:.4f}")
