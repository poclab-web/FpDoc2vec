import pickle
import numpy as np
from rdkit.Chem import AllChem
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from FpDoc2vec.py import evaluate_category, create_lightgbm_classifier
def make_name2vector(model_path: str, df: pd.DataFrame) -> np.ndarray:
    """Convert to compound vectors using NameDoc2Vec model
    
    Args:
        model_path: Path to the saved NameDoc2Vec model file
        df: DataFrame containing compound data
        
    Returns:
        NumPy array of document vectors with shape (len(df), vector_size)
    """
    model = Doc2Vec.load(model_path)
    vec = np.array([model.dv.vectors[i] for i in range(len(df))])
    return vec
    
    
def main():
    # Load dataset
    with open("../../data/10genre_dataset.pkl", "rb") as f:
        df = pickle.load(f)
    
    # Define categories to evaluate
    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin', 
        'flavouring_agent', 'agrochemical', 'volatile_oil', 'antibacterial_agent', 'insecticide'
    ]
    
    # loading Doc2Vec model
    model = Doc2Vec.load("../../model/namedoc2vec.model")
    
    # Generate compound vectors
    X_vec = np.array([model.dv.vectors[i] for i in range(len(df))])
    
    # Create classifier
    lightgbm_model = create_lightgbm_classifier()
    
    # Evaluate each category
    results = {}
    for category in categories:
        y = np.array([1 if i == category else 0 for i in df[category]])
        results[category] = evaluate_category(category, X_vec, y, lightgbm_model)
    
if __name__ == "__main__":
    main()
