from typing import List, Dict, Any,
from optuna_doc2vec import optimize_doc2vec, create_objective, process_single_category, add_vectors, ProgressCallback
# Define categories to process
categories: List[str] = [
    'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye','toxin', 'flavouring_agent', 'agrochemical', 'volatile_oil',
    'antibacterial_agent', 'insecticide'
]

# Define hyperparameter search space
# Please change the values as you like.
params: Dict[str, Dict[str, Any]] = {
    'window': {'type': 'int', 'min': 3, 'max': 10},
    'min_alpha': {'type': 'float', 'min': 0.000001, 'max': 0.025},
    'sample': {'type': 'float', 'min': 0, 'max': 0.00001},
    'epochs': {'type': 'int', 'min': 30, 'max': 1000},
    'negative': {'type': 'int', 'min': 1, 'max': 20},
    'ns_exponent': {'type': 'float', 'min': 0, 'max': 1}
}
input_traindf_path = "train_df.pkl"

# Run optimization with the loaded data
best_params: Dict[str, Any] = optimize_doc2vec(
    input_traindf_path,
    categories=categories,
    purpose_description="description",
    dimension=100,
    params=params,
    n_trials=150
)
