import pickle
import numpy as np
from rdkit.Chem import AllChem
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from FpDoc2vec.py import evaluate_category, create_lightgbm_classifier

def make_descriptor(input_path):
    # Load dataset
    with open(input_path, "rb") as f:
        df = pickle.load(f)
    # Generate compound vectors
    desc = np.array(df.iloc[:, 14:])
    return desc
    
def main():
    # Load dataset
    with open("../../10genre_32descriptor.pkl", "rb") as f:
        df = pickle.load(f)
    
    # Define categories to evaluate
    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin', 
        'flavouring_agent', 'agrochemical', 'volatile_oil', 'antibacterial_agent', 'insecticide'
    ]
    
    # Generate compound vectors
    desc = np.array(df.iloc[:, 14:])
    
    # Create classifier
    lightgbm_model = create_lightgbm_classifier()
    
    # Evaluate each category
    results = {}
    for category in categories:
        y = np.array([1 if i == category else 0 for i in df[category]])
        results[category] = evaluate_category(category, desc, y, lightgbm_model)
    
if __name__ == "__main__":
    main()
