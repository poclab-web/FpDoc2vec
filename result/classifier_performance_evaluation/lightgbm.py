import pickle
import numpy as np
import lightgbm as lgb
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import f1_score
from logistic_regression.py import addvec



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
