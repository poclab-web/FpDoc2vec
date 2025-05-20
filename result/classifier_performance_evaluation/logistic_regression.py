import pickle
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

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
    """Train and evaluate LightGBM model for a specific category
    
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
