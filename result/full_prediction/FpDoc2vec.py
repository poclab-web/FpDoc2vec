import pickle
import numpy as np
from rdkit.Chem import AllChem
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def add_vec(fingerprint_list, model):
    """
    Generate compound vectors by combining fingerprints with doc2vec model
    """
    compound_vec = []
    for i in fingerprint_df:
        fingerprint_vec = 0
        for j in i:
            fingerprint_vec += model.dv.vectors[j]
        compound_vec.append(fingerprint_vec)
        
    return compound_vec

def evaluate_category(category, X_vec, y, lightgbm_model):
    """
    Evaluate model performance for a specific category using cross-validation
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

def create_lightgbm_classifier():
    """
    Create and configure LightGBM classifier with optimized parameters
    """
    return lgb.LGBMClassifier(
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


def main():
    # Load dataset
    with open("../../data/10genre_dataset.pkl", "rb") as f:
        df = pickle.load(f)
        
    finger_list = list(df["fp_3_4096"])
    
    # Define categories to evaluate
    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin', 
        'flavouring_agent', 'agrochemical', 'volatile_oil', 'antibacterial_agent', 'insecticide'
    ]
    
    # loading Doc2Vec model
    model = Doc2Vec.load("../../model/fpdoc2vec4096.model")
    
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
