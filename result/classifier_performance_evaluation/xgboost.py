import pickle
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from xgboost import XGBClassifier as xgb
from sklearn.metrics import f1_score
from logistic_regression.py import addvec

def evaluate_category_xgboost(category, X_train, X_test, train_df, test_df):
    """
    Train and evaluate XGBoost model for a specific category
    
    Args:
        category: Category name to evaluate
        X_train: Training feature vectors
        X_test: Testing feature vectors
        train_df: Training dataframe
        test_df: Testing dataframe
        
    Returns:
        Tuple of (training_f1_score, test_f1_score)
    """
    print(f"## {category} ##")
    
    # Prepare labels
    y_train = np.array([1 if i == category else 0 for i in train_df[category]])
    y_test = np.array([1 if i == category else 0 for i in test_df[category]])
    
    # Train model with optimized hyperparameters
    xg_boost = xgb(n_estimators=375, max_depth=8,
                   learning_rate=0.036865217123129936, booster="gbtree",
                   gamma=0.9396457047650646, min_child_weight=3.567611062612357,
                   max_delta_step=4.897284214673892, subsample=0.54928595002729,
                   colsample_bytree=0.7402540955402481, reg_alpha=1,
                   reg_lambda=3, scale_pos_weight=9.305137370400152, random_state=50)
    
    xg_boost.fit(X_train, y_train)

    # Calculate scores
    y_train_pred = xg_boost.predict(X_train)
    y_test_pred = xg_boost.predict(X_test)
    
    train_score = f1_score(y_train, y_train_pred)
    test_score = f1_score(y_test, y_test_pred)
    
    # Print results
    print(f"Training Data: {train_score}")
    print(f"Test Data: {test_score}")
    
    return train_score, test_score

def main():
    """
    Main function to load data, prepare features, and evaluate XGBoost models
    for different chemical categories
    """
    # Load data
    with open("../../data/test_df.pkl", "rb") as f:
        test_df = pickle.load(f)
    with open("../../data/train_df.pkl", "rb") as f:
        train_df = pickle.load(f)
    
    # Load model
    model = Doc2Vec.load("../../model/fpdoc2vec4096_novel.model")
    
    # Define categories
    categories = ['antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin', 
                  'flavouring_agent', 'agrochemical', 'volatile_oil', 
                  'antibacterial_agent', 'insecticide']
    
    # Prepare feature vectors
    train_finger_list = list(train_df["fp_3_4096"])
    test_finger_list = list(test_df["fp_3_4096"])
    
    train_compound_vec = addvec(train_finger_list, model)
    test_compound_vec = addvec(test_finger_list, model)
    
    X_train_vec = np.array([train_compound_vec[i] for i in range(len(train_df))])
    X_test_vec = np.array([test_compound_vec[i] for i in range(len(test_df))])
    
    # Evaluate each category
    xg_train, xg_test = [], []
    for category in categories:
        train_score, test_score = evaluate_category_xgboost(
            category, X_train_vec, X_test_vec, train_df, test_df
        )
        xg_train.append(train_score)
        xg_test.append(test_score)
    
    # Calculate and print average scores
    print("\n## Average scores across all categories ##")
    print(f"Average Training F1: {np.mean(xg_train):.4f}")
    print(f"Average Test F1: {np.mean(xg_test):.4f}")

if __name__ == "__main__":
    main()
