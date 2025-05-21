from optuna_predictor_framework import process_single_category, create_objective_function, ProgressCallback, optimize_model, run_optimization, prepare_data, create_lightgbm_model, create_lightgbm_params, create_adaboost_model, create_adaboost_params,  create_rf_model, create_rf_params, create_xgboost_model, create_xgboost_params, create_et_model, create_et_params,  

# Prepare data
# Example usage - replace with your actual file paths
test_data_path = "test_df.pkl"
train_data_path = "train_df.pkl"
model_path = "fpdoc2vec.model"
train_df, test_df, X_train_vec, X_test_vec, categories = prepare_data(
    test_path=test_data_path,
    train_path=train_data_path,
    model_path=model_path
)

# Run optimization
# The following is that provides examples of LightGBM.
best_params = run_optimization(
    model_creator=create_lightgbm_model, # Please appropriately change this part.
    param_creator=create_lightgbm_params, # Please appropriately change this part.
    train_df=train_df,
    test_df=test_df,
    X_train_vec=X_train_vec,
    X_test_vec=X_test_vec,
    categories=categories,
    n_trials=300  # You can adjust the number of trials
)
