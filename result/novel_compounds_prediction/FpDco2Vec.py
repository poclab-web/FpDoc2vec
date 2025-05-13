import pickle
import numpy as np
import lightgbm as lgb
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import f1_score

def add_vec(fingerprint_list, model):
    """
    Generate compound vectors by combining fingerprints with doc2vec model
    """
    compound_vec = []
    for i in fingerprint_list:
        fingerprint_vec = 0
        for j in i:
            fingerprint_vec += model.dv.vectors[j]
        compound_vec.append(fingerprint_vec)
        
    return compound_vec

def load_data():
    """
    Load training and test data
    """
    with open("../../data/test_df.pkl", "rb") as f:
        test_df = pickle.load(f)
    with open("../../data/train_df.pkl", "rb") as f:
        train_df = pickle.load(f)
    
    return train_df, test_df

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

def train_and_evaluate_model(train_df, test_df, X_train, X_test, category, lightgbm):
    """
    Train and evaluate LightGBM model for multiple categories
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


def main():
    # Load dataset
    train_df, test_df = load_data()
        
    train_finger_list = list(train_df["fp_3_4096"])
    test_finger_list = list(test_df["fp_3_4096"])
    
    # Define categories to evaluate
    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin', 
        'flavouring_agent', 'agrochemical', 'volatile_oil', 'antibacterial_agent', 'insecticide'
    ]
    
    # loading Doc2Vec model
    model = Doc2Vec.load("../../model/fpdoc2vec4096_novel.model")
    
    # Generate compound vectors
    train_compound_vec = add_vec(train_finger_list, model)
    test_compound_vec = add_vec(test_finger_list, model)
    X_train_vec = np.array([train_compound_vec[i] for i in range(len(train_df))])
    X_test_vec = np.array([test_compound_vec[i] for i in range(len(test_df))])
    
    # Create classifier
    lightgbm_model = create_lightgbm_classifier()
    
    # Evaluate each category
    results = {}
    for category in categories:
      results[category] = train_and_evaluate_model(train_df, test_df, X_train_vec, X_test_vec, category, lightgbm_model)
    
if __name__ == "__main__":
    main()
