from shap_fpdoc2vec import main_fpdoc2vec
from shap_ecfp import generate_morgan_fingerprints, create_lightgbm_classifier, main_ecfp

# Define model parameters
# Example params - replace with your actual params
gbm_params: Dict[str, Any] = {
    "boosting_type": "dart", 
    "n_estimators": 444, 
    "learning_rate": 0.07284380689492893, 
    "max_depth": 6, 
    "num_leaves": 41, 
    "min_child_samples": 21, 
    "class_weight": "balanced", 
    "reg_alpha": 1.4922729949843299, 
    "reg_lambda": 2.8809246344115778, 
    "colsample_bytree": 0.5789063337359206, 
    "subsample": 0.5230422589468584, 
    "subsample_freq": 2, 
    "drop_rate": 0.1675163179873052, 
    "skip_drop": 0.49103811434109507, 
    "objective": 'binary', 
    "random_state": 50
}

model = create_lightgbm_classifier(params)
# Example usage - replace with your actual file paths
input_path = "10genre_dataset.pkl"
# Please modify according to the purpose.
purpose="antioxidant"
output_path = "shap_ecfp_value.pkl"
target_molecule = "quercetin"

main_ecfp(input_path, purpose, model, output_path)


# Example usage - replace with your actual file paths
input_path = "10genre_dataset.pkl"
model_path = "fpdoc2vec.model"
purpose = "antioxidant" # Please modify according to the purpose.
output_path = "shap_value_fpdoc2vec.pkl"
# Example params - replace with your actual params
gbm_params: Dict[str, Any] = {
    "boosting_type": "dart", 
    "n_estimators": 444, 
    "learning_rate": 0.07284380689492893, 
    "max_depth": 6, 
    "num_leaves": 41, 
    "min_child_samples": 21, 
    "class_weight": "balanced", 
    "reg_alpha": 1.4922729949843299, 
    "reg_lambda": 2.8809246344115778, 
    "colsample_bytree": 0.5789063337359206, 
    "subsample": 0.5230422589468584, 
    "subsample_freq": 2, 
    "drop_rate": 0.1675163179873052, 
    "skip_drop": 0.49103811434109507, 
    "objective": 'binary', 
    "random_state": 50
}
main_fpdoc2vec(input_path, purpose, model_path, gbm_params, max_evals = 500000, output_path)
