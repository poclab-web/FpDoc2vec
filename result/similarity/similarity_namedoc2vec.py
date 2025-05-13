import pickle
from gensim.models.doc2vec import Doc2Vec

def find_similar_terms(model, query_term, top_n=10):
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

def main():
    """Load data and model, then demonstrate similarity search."""
    # Load dataset
    with open("../../data/10genre_dataset.pkl", "rb") as f:
        df = pickle.load(f)
    
    # Load pre-trained Doc2Vec model
    model = Doc2Vec.load("../../model/namedoc2vec.model")
    
    # Find terms similar to "sucrose"
    similar_terms = find_similar_terms(model, "sucrose", 10)
    
    # Display results
    for term, score in similar_terms:
        print(f"  {term}: {score:.4f}")

if __name__ == "__main__":
    main()
