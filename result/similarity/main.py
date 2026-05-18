from similarity_core import get_categories, cosine_similarity, calculate_similarities, similarity_output, calculate_tanimoto_similarities, tanimoto_similarity_output
from utils import load_pickle, generate_ecfp_fingerprints, fingerprints_to_vectors
from gensim.models.doc2vec import Doc2Vec

def main():
    # Example usage - replace with your actual file paths
    input_path = "10genre_dataset.pkl"
    fpdoc2vec_model_path = "fpdoc2vec.model"
    namedoc2vec_model_path = "namedoc2vec.model"
    df = load_pickle(input_path)
    # Please change the values as you like.
    n = 3  
    finger_print, bit_list = generate_ecfp_fingerprints(df["ROMol"], 3, 4096)

    # Compound Similarity Using FpDoc2Vec Model
    fp_model = Doc2Vec.load(fpdoc2vec_model_path)
    vec = fingerprints_to_vectors(bit_list, fp_model)
    similarity_output(df, vec, "quercetin", n=n)


    # Compound Similarity Using NameDoc2Vec Model
    name_model = Doc2Vec.load(namedoc2vec_model_path)
    vec = make_name2vector(name_model, df)
    similarity_output(df, vec, "quercetin", n=n)

    # Compound Similarity Using Tanimoto similarity
    tanimoto_similarity_output(df, finger_print, "quercetin", n=n)

if __name__ == "__main__":
    main()
