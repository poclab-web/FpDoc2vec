import pickle
import numpy as np
from rdkit.Chem import AllChem
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def generate_morgan_fingerprints(df: pd.DataFrame, radius: int, n_bits: int) -> np.ndarray:
    """
    Generate Morgan fingerprints for molecules in the dataframe.
    
    Args:
        df: DataFrame containing RDKit molecule objects in a column named 'ROMol'
        radius: The radius of the Morgan fingerprint. Higher values capture more extended 
                connectivity information around each atom
        n_bits: The length of the bit vector. Larger values reduce the chance of bit collisions
    
    Returns:
        A numpy array containing fingerprints, where each row represents a molecule 
        and each column represents a bit in the fingerprint
    """
    fingerprints = []
    for i, mol in enumerate(df["ROMol"]):
        try:
            fingerprint = [j for j in AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)]
            fingerprints.append(fingerprint)
        except:
            print(f"Error processing molecule at index {i}")
            continue
    fingerprints = np.array(fingerprints)
    return fingerprints

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

def evaluate_category(category: str, 
                      X_vec: np.ndarray, 
                      y: np.ndarray, 
                      lightgbm_model: lgb.LGBMClassifier) -> Dict[str, Union[List[float], float]]:
    """Evaluate model performance for a specific category using cross-validation
    
    Args:
        category: Name of the category being evaluated
        X_vec: Feature matrix as numpy array containing compound vectors
        y: Target array containing binary labels for the category
        lightgbm_model: Pre-configured LightGBM classifier model
        
    Returns:
        Dictionary containing training and test scores:
            - train_scores: List of F1 scores for each fold (training data)
            - test_scores: List of F1 scores for each fold (test data)
            - mean_train: Mean F1 score across all folds (training data)
            - mean_test: Mean F1 score across all folds (test data)
    """
    print(f"## {category} ##")
    test_scores = []
    train_scores = []
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_idx, test_idx in skf.split(range(len(y)), y):
        X_train_vec, X_test_vec = X_vec[train_idx], X_vec[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        lightgbm_model.fit(X_train_vec, y_train)
        y_train_pred = lightgbm_model.predict(X_train_vec)
        y_test_pred = lightgbm_model.predict(X_test_vec)
        
        train_scores.append(f1_score(y_train, y_train_pred))
        test_scores.append(f1_score(y_test, y_test_pred))
    
    print(f"Training Data: {np.mean(train_scores)}")
    print(f"Test Data: {np.mean(test_scores)}")
    
    return {
        'train_scores': train_scores,
        'test_scores': test_scores,
        'mean_train': np.mean(train_scores),
        'mean_test': np.mean(test_scores)
    }

def build_doc2vec_model(corpus, list, doc2vec_param):
    """
    Build and train a Doc2Vec model from corpus and structure infomation
    """
    tagged_documents = [
        TaggedDocument(words=corpus, tags=list[i]) 
        for i, corpus in enumerate(corpus)
    ]
    
    model = Doc2Vec(tagged_documents, **doc2vec_param)
    
    return model

def main(input_path):
    # Load dataset
    with open(input_path, "rb") as f:
        df = pickle.load(f)
        
    # Generate fingerprints
    df["fp_2_2048"] = generate_morgan_fingerprints(df)
    finger_list = list(df["fp_2_2048"])
    
    # Define categories to evaluate
    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin', 
        'flavouring_agent', 'agrochemical', 'volatile_oil', 'antibacterial_agent', 'insecticide'
    ]
    
    # Prepare corpus for Doc2Vec
    corpus = [sum(doc, []) for doc in df["description_remove_stop_words"]]
    
    # Build Doc2Vec model
    model = build_doc2vec_model(corpus, finger_list)
    
    # Generate compound vectors
    compound_vec = add_vec(finger_list, model)
    X_vec = np.array([compound_vec[i] for i in range(len(df))])
    
    # Create classifier
    lightgbm_model = create_lightgbm_classifier()
    
    # Evaluate each category
    results = {}
    for category in categories:
        y = np.array([1 if i == category else 0 for i in df[category]])
        results[category] = evaluate_category(category, X_vec, y, lightgbm_model)
    
if __name__ == "__main__":
    main()
# Example usage - replace with your actual params
    doc2vec_param = {"vector_size": 100, 
         "min_count": 0,
         "window": 10,
         "min_alpha": 0.023491749982816976,
         "sample": 7.343338709169564e-06,
         "epochs": 859,
         "negative": 2,
         "ns_exponent": 0.8998927133390002,
         "workers": 1, 
         "seed": 100}

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
