import pickle
import numpy as np
import lightgbm as lgb
from typing import Dict, List, Any, Tuple
from gensim.models.doc2vec import Doc2Vec
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


def load_data(train_df_path: str, test_df_path: str) -> Tuple[Any, Any]:
    """Load training and test data from pickle files
    
    Args:
        train_df_path: Path to the training dataframe pickle file
        test_df_path: Path to the test dataframe pickle file
        
    Returns:
        Tuple containing (train_df, test_df)
    """
    with open(train_df_path, "rb") as f:
        test_df = pickle.load(f)
    with open(test_df_path, "rb") as f:
        train_df = pickle.load(f)
    
    return train_df, test_df


def train_and_evaluate_model(
    train_df: Any, 
    test_df: Any, 
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    category: str, 
    lightgbm: lgb.LGBMClassifier
) -> Dict[str, float]:
    """Train and evaluate LightGBM model for a specific category
    
    Args:
        train_df: Training dataframe containing category labels
        test_df: Test dataframe containing category labels
        X_train: Training feature matrix as numpy array
        X_test: Test feature matrix as numpy array
        category: Category name to use as target variable
        lightgbm: LightGBM classifier model instance
        
    Returns:
        Dictionary containing training and test F1 scores
    """
    y_train = np.array([1 if i == category else 0 for i in train_df[category]])
    y_test = np.array([1 if i == category else 0 for i in test_df[category]])
    
    lightgbm.fit(X_train, y_train)

    y_train_pred = lightgbm.predict(X_train)
    y_test_pred = lightgbm.predict(X_test)

    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    print(f"Training Data: {train_f1}")
    print(f"Test Data: {test_f1}")
    
    return {
        'train_scores': train_f1,
        'test_scores': test_f1
    }

def main_fpdoc2vec(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    model_path: str, 
    lightgbm_model: lgb.LGBMClassifier,
    categories: List[str]
) -> Dict[str, Dict[str, float]]:
    """Main function to run the training and evaluation process for the FpDoc2Vec method
    
    Args:
        train_df: Training DataFrame containing molecular data with 'fp_3_4096' column
        test_df: Test DataFrame containing molecular data with 'fp_3_4096' column
        model_path: Path to the saved Doc2Vec model
        lightgbm_model: Configured LightGBM classifier instance
        categories: List of category names to evaluate
        
    Returns:
        Dictionary mapping categories to their training and test scores
    """
    # Generate fingerprint lists
    train_finger_list = list(train_df["fp_3_4096"])
    test_finger_list = list(test_df["fp_3_4096"])
    
    # Loading Doc2Vec model
    model = Doc2Vec.load(model_path)
    
    # Generate compound vectors
    train_compound_vec = add_vectors(train_finger_list, model)  
    test_compound_vec = add_vectors(test_finger_list, model)   
    X_train_vec = np.array([train_compound_vec[i] for i in range(len(train_df))])
    X_test_vec = np.array([test_compound_vec[i] for i in range(len(test_df))])
    
    # Evaluate each category
    results = {}
    for category in categories:
        results[category] = train_and_evaluate_model(
            train_df, test_df, X_train_vec, X_test_vec, category, lightgbm_model
        )
    
    return results
