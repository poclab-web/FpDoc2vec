from typing import Dict, Any

dt = DecisionTreeClassifier(**dt_params)
dt_params: Dict[str, Any] = {
    "max_depth": 2, 
    "class_weight": "balanced"
}

ada_params: Dict[str, Any] = {
    "estimator": dt,
    "n_estimators": 442,
    "learning_rate": 0.07760379807069998,
    "algorithm": "SAMME.R",
    "random_state": 50
}
