from result.similarity.similarity_core import similarity_output, tanimoto_similarity_output
from utils import load_pickle, generate_ecfp_fingerprints, fingerprints_to_vectors, make_name2vector
from gensim.models.doc2vec import Doc2Vec

def main():
    """Compute and display top-n similar compounds for FpDoc2Vec, NameDoc2Vec, and Tanimoto similarity."""
    # Example usage - replace with your actual file paths
    input_path = "data/created_dataset/train_df.pkl"
    fpdoc2vec_model_path = "model/Doc2Vec_training/fpdoc2vec.model"
    namedoc2vec_model_path = "model/Doc2Vec_training/namedoc2vec.model"
    target_compound = "quercetin"  
    n = 3  

    df = load_pickle(input_path)
    finger_print, bit_list = generate_ecfp_fingerprints(list(df["ROMol"]), 3, 4096)

    # Compound Similarity Using FpDoc2Vec Model
    fp_model = Doc2Vec.load(fpdoc2vec_model_path)
    vec = fingerprints_to_vectors(bit_list, fp_model)
    similarity_output(df, vec, target_compound, n=n)

    # Compound Similarity Using NameDoc2Vec Model
    name_model = Doc2Vec.load(namedoc2vec_model_path)
    vec = make_name2vector(name_model, df)
    similarity_output(df, vec, target_compound, n=n)

    # Compound Similarity Using Tanimoto similarity
    tanimoto_similarity_output(df, finger_print, target_compound, n=n)

if __name__ == "__main__":
    main()
