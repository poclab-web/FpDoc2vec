from typing import Dict
import pandas as pd
import numpy as np
from visualization_function import main
from utils import load_data, generate_ecfp_fingerprints, make_descriptor, make_name2vector, fingerprints_to_vectors
from gensim.models import Doc2Vec

def main():

    input_path = "data/created_dataset/train_df.pkl"

    fp_model_path = "model/Doc2Vec_training/fpdoc2vec.model"
    name_model_path = "namedoc2vec.model"

    fpdoc_output_path = "fpdoc2vec_umap.png"
    namedoc_output_path = "namedoc2vec_umap.png"
    ecfp_output_path = "ecfp_umap.png"
    desc_output_path = "desc_umap.png"


    df = load_data(input_path)
    fp, bit_list = generate_ecfp_fingerprints(df["ROMol"], radius=3, n_bits=4096)

    # FP Doc2Vec
    fp_model = Doc2Vec.load(fp_model_path)
    fp_vec = fingerprints_to_vectors(bit_list, fp_model)
    main(df, fp_vec, fpdoc_output_path)

    # NameDoc2vec
    name_model = Doc2Vec.load(name_model_path)
    name_vec = make_name2vector(name_model, df)
    main(df, name_vec, namedoc_output_path)

    # ECFP
    main(df, fp, ecfp_output_path)

    # Descriptors
    desc = make_descriptor(df)    
    main(df, desc, desc_output_path)

if __name__=="__main__":
    main()







