import pickle
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from logistic_regression.py import addvec

def evaluate_category_with_rf(category, X_train, X_test, train_df, test_df):
    """
    Train and evaluate Random Forest model for a specific category
    
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
    rf = RandomForestClassifier(
        n_estimators=1075, 
        criterion="gini", 
        max_depth=16, 
        min_samples_split=30, 
        min_samples_leaf=8, 
        min_weight_fraction_leaf=0.011003051241804895, 
        max_features="sqrt", 
        bootstrap=True, 
        class_weight="balanced", 
        max_samples=0.6800609792930714, 
        random_state=50
    )
    rf.fit(X_train, y_train)

    # Calculate scores
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    
    train_score = f1_score(y_train, y_train_pred)
    test_score = f1_score(y_test, y_test_pred)
    
    # Print results
    print(f"Training Data: {train_score}")
    print(f"Test Data: {test_score}")
    
    return train_score, test_score

def main():
    """
    Main function to load data, prepare features, and evaluate Random Forest models
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
    
    # Evaluate each category with Random Forest
    rf_train, rf_test = [], []
    for category in categories:
        train_score, test_score = evaluate_category_with_rf(
            category, X_train_vec, X_test_vec, train_df, test_df
        )
        rf_train.append(train_score)
        rf_test.append(test_score)
    
    # Calculate and print average scores
    print("\n## Average scores across all categories ##")
    print(f"Average Training F1: {np.mean(rf_train):.4f}")
    print(f"Average Test F1: {np.mean(rf_test):.4f}")

if __name__ == "__main__":
    main()
