import pickle
import numpy as np
import lightgbm as lgb
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import f1_score
from rdkit.Chem import AllChem
from FpDoc2Vec.py import load_data, create_lightgbm_classifier, train_and_evaluate_model

def fin(df):
  """
  Generate ECFP
  """
    fingerprints = []
    for i, mol in enumerate(df["ROMol"]):
        try:
            fingerprint = [j for j in AllChem.GetMorganFingerprintAsBitVect(mol, 3, 4096)]
            fingerprints.append(fingerprint)
        except:
            print("Error", i)
            continue
    fingerprints = np.array(fingerprints)
    return fingerprints

def main():
    # Load dataset
    with open("../../data/10genre_32descriptor.pkl", "rb") as f:
      df = pickle.load(f)
    train_df, test_df = load_data()
        
    train_desc = np.array(fin(train_df))
    test_desc = np.array(fin(test_df))
    
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
      results[category] = train_and_evaluate_model(train_df, test_df, train_desc, test_desc, category, lightgbm_model)
    
if __name__ == "__main__":
    main()
