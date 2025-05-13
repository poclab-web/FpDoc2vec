import pickle
import numpy as np
import lightgbm as lgb
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import f1_score

def addvec(fingerprint_list, model):
    """
    Convert fingerprints to vector representation using Doc2Vec model
    
    Args:
        fingerprint_list: List of molecular fingerprints
        model: Trained Doc2Vec model
        
    Returns:
        List of compound vectors
    """
    compound_vectors = []
    for fp in fingerprint_list:
        compound_vectors.append(model.infer_vector(fp))
    return compound_vectors

def evaluate_category_lightgbm(category, X_train, X_test, train_df, test_df):
    """
    Train and evaluate LightGBM model for a specific category
    
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
    lightgbm = lgb.LGBMClassifier(
        boosting_type="dart", 
        n_estimators=444, 
        learning_rate=0.07284380689492893, 
        max_depth=6, 
        num_leaves=41, 
        min_child_samples=21, 
        class_weight="balanced", 
        reg_alpha=1.4922729949843299, 
        reg_lambda=2.8809246344115778, 
        colsample_bytree=0.5789063337359206, 
        subsample=0.5230422589468584, 
        subsample_freq=2, 
        drop_rate=0.1675163179873052, 
        skip_drop=0.49103811434109507, 
        objective='binary', 
        random_state=50
    )
    lightgbm.fit(X_train, y_train)

    # Calculate scores
    y_train_pred = lightgbm.predict(X_train)
    y_test_pred = lightgbm.predict(X_test)
    
    train_score = f1_score(y_train, y_train_pred)
    test_score = f1_score(y_test, y_test_pred)
    
    # Print results
    print(f"Training Data: {train_score}")
    print(f"Test Data: {test_score}")
    
    return train_score, test_score

def main():
    """
    Main function to load data, prepare features, and evaluate LightGBM models
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
    lightgbm_train, lightgbm_test = [], []
    for category in categories:
        train_score, test_score = evaluate_category_lightgbm(
            category, X_train_vec, X_test_vec, train_df, test_df
        )
        lightgbm_train.append(train_score)
        lightgbm_test.append(test_score)
    
    # Calculate and print average scores
    print("\n## Average scores across all categories ##")
    print(f"Average Training F1: {np.mean(lightgbm_train):.4f}")
    print(f"Average Test F1: {np.mean(lightgbm_test):.4f}")

if __name__ == "__main__":
    main()
