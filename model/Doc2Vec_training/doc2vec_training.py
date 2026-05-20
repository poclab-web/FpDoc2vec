import pandas as pd
from gensim.models.doc2vec import Doc2Vec

from config import doc2vec_params as DOC2VEC_PARAMS
from utils import load_pickle, generate_ecfp_fingerprints, build_tagged_corpus

def main():
    # Example usage - replace with your actual file paths
    dataframe_path = "created_dataset/train_df.pkl"
    fpmodel_path = "fpmodel.model"
    namemodel_path = "namedoc2vec.model"
    description_column = "processed_description"


    df = load_pickle(dataframe_path)

    # 1. FpDoc2Vec (training dataset, Morgan fingerprint tags)
    finger_prints, bit_list = generate_ecfp_fingerprints(list(df["ROMol"]), radius=3, n_bits=4096)
    corpus = build_tagged_corpus(df, bit_list, description_column)
    fp_model = Doc2Vec(corpus, **DOC2VEC_PARAMS)
    fp_model.save(fpmodel_path)

    # 2. NameDoc2Vec (training dataset, compound name tags)
    name_tags = list(df["NAME"])
    corpus = build_tagged_corpus(df, name_tags, description_column)
    name_model = Doc2Vec(corpus, **DOC2VEC_PARAMS)
    name_model.save(namemodel_path)

if __name__ == "__main__":
    main()
