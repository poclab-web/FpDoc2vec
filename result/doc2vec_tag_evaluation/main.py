import numpy as np
import lightgbm as lgb
from gensim.models import Doc2Vec

from config.lightgbm_params import gbm_params as LIGHTGBM_PARAMS
from config.doc2vec_params import doc2vec_param as DOC2VEC_PARAMS
from utils import (
    build_tagged_corpus,
    generate_ecfp_fingerprints,
    fingerprints_to_vectors,
    load_pickle,
    save_pickle,
    main_cv,
)
from result.doc2vec_tag_evaluation.core import (
    run_ecfp,
    run_with_filter,
    generate_maccs_on_bits,
    generate_pharmacophore_on_bits,
    generate_ngram_indices,
)

def main():

    DATA_PATH = "data/created_dataset/train_df.pkl"
    DESCRIPTION_COL = "description_gensim"

    ecfp_2048_model_path = "results/doc2vec_tag_evaluation/model_ecfp2048.pkl"
    ecfp_4096_model_path = "results/doc2vec_tag_evaluation/model_ecfp4096.pkl"
    maccs_model_path = "results/doc2vec_tag_evaluation/model_maccs.pkl"
    pharmacophore_model_path = "results/doc2vec_tag_evaluation/model_pharmacophore.pkl"
    ngram_model_path = "results/doc2vec_tag_evaluation/model_ngram.pkl"

    ecfp_2048results_path = "results/doc2vec_tag_evaluation/ecfp2048.pkl"
    ecfp_4096results_path = "results/doc2vec_tag_evaluation/ecfp4096.pkl"
    maccs_result_path = "results/doc2vec_tag_evaluation/maccs.pkl"
    pharmacophore_result_path = "results/doc2vec_tag_evaluation/pharmacophore.pkl"
    ngram_result_path = "results/doc2vec_tag_evaluation/smiles_ngram.pkl"

    df = load_pickle(DATA_PATH)
    lgbm = lgb.LGBMClassifier(**LIGHTGBM_PARAMS)

    # ECFP (radius=2, 2048 bits)
    _, on_bits_ecfp2048 = generate_ecfp_fingerprints(list(df["ROMol"]), radius=2, n_bits=2048)
    corpus_ecfp2048 = build_tagged_corpus(df, on_bits_ecfp2048, DESCRIPTION_COL)
    model_ecfp2048 = Doc2Vec(corpus_ecfp2048, **DOC2VEC_PARAMS)
    model_ecfp2048.save(ecfp_2048_model_path)
    results_ecfp2048 = run_ecfp(df, on_bits_ecfp2048, lgbm, model_ecfp2048)
    save_pickle(results_ecfp2048, ecfp_2048results_path)

    # ECFP (radius=3, 4096 bits)
    _, on_bits_ecfp4096 = generate_ecfp_fingerprints(list(df["ROMol"]), radius=3, n_bits=4096)
    corpus_ecfp4096 = build_tagged_corpus(df, on_bits_ecfp4096, DESCRIPTION_COL)
    model_ecfp4096 = Doc2Vec(corpus_ecfp4096, **DOC2VEC_PARAMS)
    model_ecfp4096.save(ecfp_4096_model_path)
    results_ecfp4096 = run_ecfp(df, on_bits_ecfp4096, lgbm, model_ecfp4096)
    save_pickle(results_ecfp4096, ecfp_4096results_path)

    # MACCS keys
    on_bits_maccs, invalid_maccs = generate_maccs_on_bits(df)
    valid_mask_maccs = np.array([b is not None for b in on_bits_maccs])
    corpus_maccs = build_tagged_corpus(
        df[valid_mask_maccs].reset_index(drop=True),
        [b for b in on_bits_maccs if b is not None],
        DESCRIPTION_COL,
    )
    model_maccs = Doc2Vec(corpus_maccs, **DOC2VEC_PARAMS)
    model_maccs.save(maccs_model_path)
    results_maccs = run_with_filter(df, on_bits_maccs, invalid_maccs, lgbm, model_maccs)
    save_pickle(results_maccs, maccs_result_path)

    # 2D pharmacophore (Gobbi)
    on_bits_pharm, invalid_pharm = generate_pharmacophore_on_bits(df)
    valid_mask_pharm = np.array([b is not None for b in on_bits_pharm])
    corpus_pharm = build_tagged_corpus(
        df[valid_mask_pharm].reset_index(drop=True),
        [b for b in on_bits_pharm if b is not None],
        DESCRIPTION_COL,
    )
    model_pharm = Doc2Vec(corpus_pharm, **DOC2VEC_PARAMS)
    model_pharm.save(pharmacophore_model_path)
    results_pharm = run_with_filter(df, on_bits_pharm, invalid_pharm, lgbm, model_pharm)
    save_pickle(results_pharm, pharmacophore_result_path)

    # SMILES character 3-grams
    ngram_indices = generate_ngram_indices(df, smiles_col="smiles", n=3)
    corpus_ngram = build_tagged_corpus(df, ngram_indices, DESCRIPTION_COL)
    model_ngram = Doc2Vec(corpus_ngram, **DOC2VEC_PARAMS)
    model_ngram.save(ngram_model_path)
    X_ngram = fingerprints_to_vectors(ngram_indices, model_ngram)
    results_ngram = main_cv(df, X_ngram, lgbm)
    save_pickle(results_ngram, ngram_result_path)

if __name__ == "__main__":
    main()