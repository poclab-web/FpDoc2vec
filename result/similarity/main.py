from similarity_namedoc2vec import find_similar_terms, similarity_namedoc2vec
from similarity_fpdoc2vec import cosine_similarity, calculate_similarities, get_top_similar_compounds, add_vectors, similarity_fpdoc2vec
from similarity_tanimoto import calculate_tanimoto_similarities, get_top_similar_compounds, similarity_tanimoto


# Example usage - replace with your actual file paths
input_path = "10genre_dataset.pkl"
n = 10  # Please change the values as you like.

# Compound Similarity Using FpDoc2Vec Model
model_path_fp = "fpdoc2vec.model"
similarity_fpdoc2vec(input_path, model_path_fp, n=n)

# Compound Similarity Using NameDoc2Vec Model
model_path_name = "namedoc2vec.model"
similarity_namedoc2vec(input_path, model_path_name, n=n)

# Compound Similarity Using Tanimoto similarity
similarity_tanimoto(input_path, n=n)
