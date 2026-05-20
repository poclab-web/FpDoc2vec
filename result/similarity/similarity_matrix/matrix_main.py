from gensim.models import Doc2Vec
from utils import load_pickle, save_pickle, generate_ecfp_fingerprints, fingerprints_to_vectors, make_name2vector, make_descriptor
from result.similarity.similarity_matrix.matrix_core import export_upper_triangle_matrix, compare_matrices
from sklearn.metrics.pairwise import cosine_similarity


def main():
    """Compute similarity matrices for FpDoc2Vec, NameDoc2Vec, ECFP, and descriptors, then compare them pairwise."""
    input_path = "data/created_dataset/train_df.pkl"
    input_descriptor_path = "data/created_dataset/train_desc_df.pkl"
    fpmodel_path = "model/Doc2Vec_training/fpdoc2vec.model"
    namemodel_path = "model/Doc2Vec_training/namedoc2vec.model"

    result_fpdoc_path = "fpdoc_similarity_matrix.xlsx"
    result_namedoc_path = "namedoc_similarity_matrix.xlsx"
    result_ecfp_path = "ecfp_similarity_matrix.xlsx"
    result_descriptor_path = "descriptor_similarity_matrix.xlsx"

    df = load_pickle(input_path)
    desc_df = load_pickle(input_descriptor_path)
    finger_print, bit_list = generate_ecfp_fingerprints(list(df["ROMol"]), 3, 4096)

    #fp doc2vec
    fp_model = Doc2Vec.load(fpmodel_path)
    fpvec = fingerprints_to_vectors(bit_list, fp_model)
    fpsimilarity_matrix = cosine_similarity(fpvec)

    #name doc2vec
    name_model = Doc2Vec.load(namemodel_path)
    namevec = make_name2vector(model=name_model, df=df)
    namesimilarity_matrix = cosine_similarity(namevec)


    #ECFP
    ecfpsimilarity_matrix = cosine_similarity(finger_print)

    #descriptors
    desc = make_descriptor(desc_df)
    desc_similarity_matrix = cosine_similarity(desc)

    # Export similarity matrices
    fpdoc_matrix = export_upper_triangle_matrix(fpsimilarity_matrix, result_fpdoc_path)
    namedoc_matrix = export_upper_triangle_matrix(namesimilarity_matrix, result_namedoc_path)
    ecfp_matrix = export_upper_triangle_matrix(ecfpsimilarity_matrix, result_ecfp_path)
    descriptor_matrix = export_upper_triangle_matrix(desc_similarity_matrix, result_descriptor_path)
    frobenius_matrix, pearson_matrix, spearman_matrix = compare_matrices(fpdoc_matrix, namedoc_matrix, ecfp_matrix, descriptor_matrix)

    save_pickle(frobenius_matrix, "frobenius_matrix.pkl")
    save_pickle(pearson_matrix, "pearson_matrix.pkl")
    save_pickle(spearman_matrix, "spearman_matrix.pkl")

if __name__ == "__main__":
    main()
