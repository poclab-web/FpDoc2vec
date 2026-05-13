from gensim.models.doc2vec import Doc2Vec
from Descriptors import make_descriptor
from config import gbm_params as params
import lightgbm as lgb
from utils import load_data, save_pickle, generate_ecfp_fingerprints, main_cv
from features import make_fp2vector, make_name2vector

def main():

    # Load dataset
    # Example paths - replace with actual paths
    input_path = "data/created_dataset/train_df.pkl"
    fp_model_path = "model/Doc2Vec_training/fpdoc2vec.model"
    name_model_path = "namedoc2vec.model"
    input_descriptor_path = "data/Descriptor/train_desc_df.pkl"

    fp_results_path = "results/full_prediction/fp_results.pkl"
    name_results_path = "results/full_prediction/name_results.pkl"
    ecfp_results_path = "results/full_prediction/ecfp_results.pkl"
    desc_results_path = "results/full_prediction/desc_results.pkl"

    df = load_data(input_path)
    classifier = lgb.LGBMClassifier(**params)

    # FP Doc2Vec approach
    fp_model = Doc2Vec.load(fp_model_path)
    fpvec = make_fp2vector(model=fp_model, df=df)
    fp_results = main_cv(df=df, X_vec=fpvec, classifier=classifier)
    save_pickle(fp_results, fp_results_path)

    # Name Doc2Vec approach
    name_model = Doc2Vec.load(name_model_path)
    namevec = make_name2vector(model=name_model, df=df)
    name_results = main_cv(df=df, X_vec=namevec, classifier=classifier)
    save_pickle(name_results, name_results_path)

    # ECFP approach
    ecfp = generate_ecfp_fingerprints(df, radius=3,n_bits=4096)[0]
    ecfp_results = main_cv(df=df, X_vec=ecfp, classifier=classifier)
    save_pickle(ecfp_results, ecfp_results_path)

    # Descriptor approach
    desc_df = load_data(input_descriptor_path)
    desc = make_descriptor(desc_df)
    desc_results = main_cv(df=df, X_vec=desc, classifier=classifier)
    save_pickle(desc_results, desc_results_path)

if __name__ == "__main__":
    main()