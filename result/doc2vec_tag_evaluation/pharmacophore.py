import pickle
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from ECFP2048bit.py import add_vec, build_doc2vec_model, create_lightgbm_classifier
from MACCSkeys import create_index_mapping

def process_pharmacophore_features(df):
    """
    Process pharmacophore features and identify invalid entries
    """
    pharmacore_list = []
    for i in tqdm(df["ROMol"]):
        try:
            fp = Generate.Gen2DFingerprint(i,Gobbi_Pharm2D.factory)
            fp_bits = list(fp.GetOnBits())
            if len(fp_bits) == 0:
                pharmacore_list.append(None)
            else:
                pharmacore_list.append(fp_bits)
        except:
            print("Error")
            pharmacore_list.append(None)
            
    invalid_pharmacore_indices = []
    
    for idx, feature in enumerate(pharmacore_list):
        if feature is None:
            invalid_pharmacore_indices.append(idx)
            
    return pharmacore_list, invalid_pharmacore_indices

def main():
    # Load dataset
    with open("../../data/10genre_dataset.pkl", "rb") as f:
        df = pickle.load(f)
    
    
    # Process pharmacophore features
    pharmacore_list, invalid_pharmacore_indices = process_pharmacophore_features(df)
    df["pharmacore"] = pharmacore_list
    
    # Create a filtered dataframe with valid pharmacophore features
    df_pharm = df.copy()
    df_pharm = df_pharm.drop(invalid_pharmacore_indices).reset_index(drop=True)
    print(f"Number of compounds with valid pharmacophore features: {len(df_pharm)}")
    
    # Prepare data for Doc2Vec
    pharm_list = list(df_pharm["pharmacore"])
    corpus = [sum(doc, []) for doc in df_pharm["description_remove_stop_words"]]
    
    # Define categories
    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin',
        'flavouring_agent', 'agrochemical', 'volatile_oil', 'antibacterial_agent', 'insecticide'
    ]
    
    # Build Doc2Vec model
    model = build_doc2vec_model(corpus, pharm_list)
    
    # Generate compound vectors
    compound_vec = add_vec(pharm_list, model)
    X_vec_pharm = np.array([compound_vec[i] for i in range(len(df_pharm))])
    
    # Create index mapping
    index_mapping = create_index_mapping(len(df), invalid_pharmacore_indices)
    
    # Evaluate model performance
    results = evaluate_with_pharmacophore(df, df_pharm, X_vec_pharm, categories, index_mapping)
    
if __name__ == "__main__":
    main()
