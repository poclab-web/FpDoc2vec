
from config import lightgbm_params as params
from utils import load_data, save_pickle, generate_ecfp_fingerprints, fingerprints_to_vectors, main_traintest, make_descriptor
from gensim.models import Doc2Vec
import lightgbm as lgb

def main() :
    """Evaluate FpDoc2Vec, ECFP, and descriptor features on unseen compounds using a train/test split."""
    # Example paths - replace with actual paths
    train_df_path = "data/created_dataset/train_df.pkl"
    test_df_path = "data/created_dataset/test_df.pkl"
    train_desc_df = "data/Descriptor/train_desc_df.pkl"
    test_desc_df = "data/Descriptor/test_desc_df.pkl"
    fpmodel_path = "model/Doc2Vec_training/fpdoc2vec.model"

    fp_result_pathname = "fpdoc2vec_results.pkl" 
    ecfp_result_pathname = "ecfp_results.pkl"
    descriptor_result_pathname = "descriptors_results.pkl"

    # Load data
    train_df, test_df = load_data(train_df_path), load_data(test_df_path)
    train_desc_df, test_desc_df = load_data(train_desc_df), load_data(test_desc_df)
    train_fp, train_bit_list = generate_ecfp_fingerprints(list(train_df["ROMol"]), radius=3, n_bits=4096)
    test_fp, test_bit_list = generate_ecfp_fingerprints(list(test_df["ROMol"]), radius=3, n_bits=4096)

    # Create classifier
    lightgbm_model: lgb.LGBMClassifier = lgb.LGBMClassifier(**params)

    # Fp Doc2vec
    fp_model = Doc2Vec.load(fpmodel_path)
    X_train_vec, X_test_vec = fingerprints_to_vectors(train_bit_list, fp_model), fingerprints_to_vectors(test_bit_list, fp_model)
    fpdoc2vec_results = main_traintest(train_df, test_df, X_train_vec, X_test_vec, lightgbm_model)
    save_pickle(fpdoc2vec_results, fp_result_pathname)

    # ECFP
    ecfp_results = main_traintest(train_df, test_df, train_fp, test_fp, lightgbm_model)
    save_pickle(ecfp_results, ecfp_result_pathname)

    # Descriptiors
    X_train_desc, X_test_desc = make_descriptor(train_desc_df), make_descriptor(test_desc_df)
    descriptor_results = main_traintest(train_df, test_df, X_train_desc, X_test_desc, lightgbm_model)
    save_pickle(descriptor_results, descriptor_result_pathname)

if __name__ == "__main__":
    main()