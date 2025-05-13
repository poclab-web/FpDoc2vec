import pickle
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from ECFP2048bit import add_vec, evaluate_category, build_doc2vec_model, create_lightgbm_classifier

def smiles_to_ngrams(smiles_list, n):
    """
    Convert SMILES strings to n-grams
    
    Args:
        smiles_list: List of SMILES strings
        n: n-gram size
        
    Returns:
        List of n-grams for each SMILES
    """
    ngrams_list = []
    for smiles in smileslist:
        if len(smiles) < n:
            ngrams_list.append([smiles])
        else:
            ngrams_list.append([smiles[i:i+n] for i in range(len(smiles) - n + 1)])
    return ngrams_list
  
def main():
    # Load dataset
    with open("../../data/10genre_dataset.pkl", "rb") as f:
        df = pickle.load(f)
      
    # Generate n-grams from SMILES   
    smiles_list = list(df["smiles"])
    ngrams_list = smiles_to_ngrams(smiles_list, 3)

    # Convert n-grams to binary vectors
    vectorizer = CountVectorizer(binary=True, analyzer=lambda x: x)
    vec = vectorizer.fit_transform(ngrams_list)
  
    # Convert sparse vectors to index lists
    ngram_list = []
    for i in vec.toarray():
        li = []
        for j in range(len(i)):
            if i[j] == 1:
                li.append(j)
        ngram_list.append(li)
    
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
