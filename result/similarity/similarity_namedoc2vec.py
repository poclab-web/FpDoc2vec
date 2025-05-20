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
