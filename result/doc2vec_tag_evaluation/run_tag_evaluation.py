import lightgbm as lgb

from config.lightgbm_params import gbm_params as LIGHTGBM_PARAMS
from util import (
    build_doc2vec_model,
    fingerprints_to_vectors,
    evaluate_all_categories,
    print_mcc_summary,
)
from main import (
    load_data,
    save_results,
    run_ecfp,
    run_with_filter,
    generate_maccs_on_bits,
    generate_pharmacophore_on_bits,
    generate_ngram_indices,
)
from config.doc2vec_params import doc2vec_param as DOC2VEC_PARAMS

DATA_PATH = "../../data/created_dataset/train_df.pkl"
DESCRIPTION_COL = "description_gensim"


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    lgbm = lgb.LGBMClassifier(**LIGHTGBM_PARAMS)

    # ECFP (radius=2, 2048 bits)
    
    results_ecfp2048 = run_ecfp(df, radius=2, n_bits=2048, lgbm=lgbm, desc_col=DESCRIPTION_COL)
    save_results(results_ecfp2048, "ecfp2048.pkl")

    # ECFP (radius=3, 4096 bits)
    results_ecfp4096 = run_ecfp(df, radius=3, n_bits=4096, lgbm=lgbm, desc_col=DESCRIPTION_COL)
    save_results(results_ecfp4096, "ecfp4096.pkl")

    # MACCS keys
    on_bits_maccs, invalid_maccs = generate_maccs_on_bits(df)
    results_maccs = run_with_filter(df, on_bits_maccs, invalid_maccs, lgbm, DESCRIPTION_COL)
    save_results(results_maccs, "maccs.pkl")

    # 2D pharmacophore (Gobbi)
    on_bits_pharm, invalid_pharm = generate_pharmacophore_on_bits(df)
    results_pharm = run_with_filter(df, on_bits_pharm, invalid_pharm, lgbm, DESCRIPTION_COL)
    save_results(results_pharm, "pharmacophore.pkl")

    # SMILES character 3-grams
    ngram_indices = generate_ngram_indices(df, smiles_col="smiles", n=3)
    model_ngram = build_doc2vec_model(df[DESCRIPTION_COL].tolist(), ngram_indices, DOC2VEC_PARAMS)
    X_ngram = fingerprints_to_vectors(ngram_indices, model_ngram)
    results_ngram = evaluate_all_categories(X_ngram, df, lgbm)
    save_results(results_ngram, "smiles_ngram.pkl")
