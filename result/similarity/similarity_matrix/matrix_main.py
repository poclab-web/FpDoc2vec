from gensim.models import Doc2Vec
from utils import load_pickle, save_pickle, generate_ecfp_fingerprints
from matrix_core import export_upper_triangle_matrix, compare_matrices
from sklearn.metrics.pairwise import cosine_similarity


def main():

    input_path = "data/train_df2.pkl"
    input_descriptor_path = "data/train_desc_df.pkl"
    fpmodel_path = "train_df_doc2vec.model"
    namemodel_path = "train_df_namedoc2vec.model"

    result_fpdoc_path = "fpdoc_similarity_matrix.xlsx"
    result_namedoc_path = "namedoc_similarity_matrix.xlsx"
    result_ecfp_path = "ecfp_similarity_matrix.xlsx"
    result_descriptor_path = "descriptor_similarity_matrix.xlsx"

    df = load_pickle(input_path)
    desc_df = load_pickle(input_descriptor_path)
    finger_print, bit_list = generate_ecfp_fingerprints(df["ROMol"], 3, 4096)

    #fp doc2vec

    fp_model = Doc2Vec.load(fpmodel_path)
    fpvec = make_fp2vector(model=fp_model, df=df)
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
