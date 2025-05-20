import pickle
import numpy as np
import lightgbm as lgb
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import f1_score
from FpDoc2Vec.py import load_data, create_lightgbm_classifier, train_and_evaluate_model

def main():
    # Load dataset
    with open(input_, "rb") as f:
      df = pickle.load(f)
    train_df, test_df = load_data()
        
    test_df1 = df[df["inchikey"].isin(list(test_df["inchikey"]))]
    train_df1 = df.drop(test_df1.index)
    train_desc = np.array(train_df1.iloc[:, 14:])
    test_desc = np.array(test_df1.iloc[:, 14:])
    
    # Define categories to evaluate
    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin', 
        'flavouring_agent', 'agrochemical', 'volatile_oil', 'antibacterial_agent', 'insecticide'
    ]
    
    # Create classifier
    lightgbm_model = create_lightgbm_classifier()
    
    # Evaluate each category
    results = {}
    for category in categories:
      results[category] = train_and_evaluate_model(train_df1, test_df1, train_desc, test_desc, category, lightgbm_model)
    
if __name__ == "__main__":
    main()
