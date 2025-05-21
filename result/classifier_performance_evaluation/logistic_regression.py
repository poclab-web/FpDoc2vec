import pickle
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from typing import List, Dict, Any

def add_vectors(fp_list: List[List[int]], model: Doc2Vec) -> List[np.ndarray]:
    """Combine document vectors based on fingerprints
    
    Args:
        fp_list: List of fingerprint lists, where each fingerprint is represented as a list of indices
        model: Trained Doc2Vec model containing document vectors
        
    Returns:
        List of compound vectors as numpy arrays
    """
    compound_vec = []
    for i in fp_list:
        fingerprint_vec = 0
        for j in i:
            fingerprint_vec += model.dv.vectors[j]
        compound_vec.append(fingerprint_vec)
    return compound_vec

def train_and_evaluate_model(
    train_df: Any, 
    test_df: Any, 
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    category: str, 
    estimator
) -> Dict[str, float]:
    """Train and evaluate model for a specific category
    
    Args:
        train_df: Training dataframe containing category labels
        test_df: Test dataframe containing category labels
        X_train: Training feature matrix as numpy array
        X_test: Test feature matrix as numpy array
        category: Category name to use as target variable
        estimator: classifier model instance
        
    Returns:
        Dictionary containing training and test F1 scores
    """
    y_train = np.array([1 if i == category else 0 for i in train_df[category]])
    y_test = np.array([1 if i == category else 0 for i in test_df[category]])
    
    estimator.fit(X_train, y_train)

    y_train_pred = estimator.predict(X_train)
    y_test_pred = estimator.predict(X_test)

    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    print(f"Training Data: {train_f1}")
    print(f"Test Data: {test_f1}")
    
    return {
        'train_scores': train_f1,
        'test_scores': test_f1
    }

def main(traindf_path: str, testdf_path: str, model_path: str, estimator: Any) -> None:
    """
    Main function to load data, prepare features, and evaluate models
    for different chemical categories
    
    Args:
        traindf_path: Path to pickle file containing training DataFrame
        testdf_path: Path to pickle file containing test DataFrame
        model_path: Path to saved Doc2Vec model
        estimator: ML model object with fit and predict methods
    """
    # Load data
    with open(traindf_path, "rb") as f:
        train_df = pickle.load(f)
    with open(testdf_path, "rb") as f:
        test_df = pickle.load(f)
    
    # Load model
    model = Doc2Vec.load(model_path)
    
    # Define categories
    categories = ['antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin', 
                  'flavouring_agent', 'agrochemical', 'volatile_oil', 
                  'antibacterial_agent', 'insecticide']
    
    # Prepare feature vectors
    train_finger_list = list(train_df["fp_3_4096"])
    test_finger_list = list(test_df["fp_3_4096"])
    
    train_compound_vec = add_vectors(train_finger_list, model)
    test_compound_vec = add_vectors(test_finger_list, model)
    
    X_train_vec = np.array([train_compound_vec[i] for i in range(len(train_df))])
    X_test_vec = np.array([test_compound_vec[i] for i in range(len(test_df))])
    
    # Evaluate each category
    train_scores, test_scores = [], []
    for category in categories:
        train_score, test_score = train_and_evaluate_model(
            train_df, test_df, X_train_vec, X_test_vec, category, estimator
        )
        train_scores.append(train_score)
        test_scores.append(test_score)

    print(f"Model: {type(estimator).__name__}")
    # Calculate and print average scores
    print("\n## Average scores across all categories ##")
    print(f"Average Training F1: {np.mean(train_scores):.4f}")
    print(f"Average Test F1: {np.mean(test_scores):.4f}")
