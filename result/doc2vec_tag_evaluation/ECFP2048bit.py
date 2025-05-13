import pickle
import numpy as np
from rdkit.Chem import AllChem
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def generate_morgan_fingerprints(df):
    """
    Generate Morgan fingerprints (ECFP4) for molecules in the dataframe
    Returns a list of indices where each fingerprint bit is 1
    """
    fingerprints = []
    for i, mol in enumerate(df["ROMol"]):
        try:
            fingerprint = [j for j in AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            fingerprints.append(fingerprint)
        except:
            print("Error", i)
            continue
    fingerprints = np.array(fingerprints)
    return [[j for j in range(2048) if i[j] == 1] for i in fingerprints]

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

def build_doc2vec_model(corpus, fingerprint_list):
    """
    Build and train a Doc2Vec model from corpus and fingerprints
    """
    tagged_documents = [
        TaggedDocument(words=corpus, tags=fingerprint_list[i]) 
        for i, corpus in enumerate(corpus)
    ]
    
    model = Doc2Vec(
        tagged_documents, 
        vector_size=100, 
        min_count=0,
        window=10,
        min_alpha=0.023491749982816976,
        sample=7.343338709169564e-06,
        epochs=859,
        negative=2,
        ns_exponent=0.8998927133390002,
        workers=1, 
        seed=100
    )
    
    return model

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
    compound_vec = add_vectors(finger_list, model)
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
