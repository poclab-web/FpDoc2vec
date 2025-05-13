import pickle
import numpy as np
from rdkit.Chem import MACCSkeys
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from ECFP2048bit.py import add_vec, build_doc2vec_model, create_lightgbm_classifier

def generate_maccs_fingerprints(df):
    """
    Generate MACCS fingerprints for molecules in the dataframe
    Returns a list of fingerprints and identifies invalid molecules
    """
    maccs_features = []
    invalid_indices = []
    
    for idx, mol in enumerate(df["ROMol"]):
        fps = MACCSkeys.GenMACCSKeys(mol)
        fp_bits = list(fps.GetOnBits())
        
        if len(fp_bits) == 0:
            print(f"Invalid MACCS for index {idx}")
            maccs_features.append(None)
            invalid_indices.append(idx)
        else:
            maccs_features.append(fp_bits)
            
    return maccs_features, invalid_indices

def create_index_mapping(df_length, invalid_indices):
    """
    Create a mapping from original dataframe indices to filtered dataframe indices
    """
    original_to_filtered = {}
    filtered_idx = 0
    
    for orig_idx in range(df_length):
        if orig_idx not in invalid_indices:
            original_to_filtered[orig_idx] = filtered_idx
            filtered_idx += 1
            
    return original_to_filtered

def evaluate_with_maccs(df, df_maccs, X_vec_maccs, categories, index_mapping):
    """
    Evaluate model performance using MACCS fingerprints and Doc2Vec vectors
    """
    lightgbm_model = create_lightgbm_classifier()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    
    results = {}
    for category_idx, category in enumerate(categories):
        maccs_train_f1_list, maccs_test_f1_list = []
        
        # Prepare target variables
        y = np.array([1 if i == category else 0 for i in df[category]])
        y_maccs = np.array([1 if i == category else 0 for i in df_maccs[category]])
        
        for train_idx, test_idx in skf.split(range(len(df)), y):
            # Convert indices to MACCS-compatible indices
            train_idx_maccs = [index_mapping[idx] for idx in train_idx if idx in index_mapping]
            test_idx_maccs = [index_mapping[idx] for idx in test_idx if idx in index_mapping]
            
            # Only proceed if we have enough samples in both train and test sets
            if len(train_idx_maccs) > 0 and len(test_idx_maccs) > 0:
                # Extract training and testing data
                X_train_maccs = X_vec_maccs[train_idx_maccs]
                X_test_maccs = X_vec_maccs[test_idx_maccs]
                y_train_maccs = y_maccs[train_idx_maccs]
                y_test_maccs = y_maccs[test_idx_maccs]
                
                # Train model and make predictions
                lightgbm_model.fit(X_train_maccs, y_train_maccs)
                y_pred_train_maccs = lightgbm_model.predict(X_train_maccs)
                y_pred_test_maccs = lightgbm_model.predict(X_test_maccs)
                
                # Calculate F1 score
                maccs_train_f1_list.append(f1_score(y_train_maccs, y_pred_train_maccs))
                maccs_test_f1_list.append(f1_score(y_test_maccs, y_pred_test_maccs))
              
          print(f"Training Data: {np.mean(maccs_train_f1_list)}")
          print(f"Test Data: {np.mean(maccs_test_f1_list)}")
      
    return {
        'train_scores': maccs_train_f1_list,
        'test_scores': maccs_test_f1_list,
        'mean_train': np.mean(maccs_train_f1_list),
        'mean_test': np.mean(maccs_test_f1_list)
    }

def main():
    # Load dataset
    with open("chemdata/10genre_predict.pkl", "rb") as f:
        df = pickle.load(f)
    
    # Generate MACCS fingerprints
    maccs_features, invalid_indices = generate_maccs_fingerprints(df)
    
    # Create a filtered dataframe with valid MACCS fingerprints
    df_maccs = df.copy()
    df_maccs["maccs"] = maccs_features
    df_maccs = df_maccs.dropna(subset=['maccs'])
    print(f"Number of compounds with valid MACCS keys: {len(df_maccs)}")
    
    # Prepare data for Doc2Vec
    maccs_list = list(df_maccs["maccs"])
    corpus = [sum(doc, []) for doc in df_maccs["description_remove_stop_words"]]
    
    # Define categories
    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin',
        'flavouring_agent', 'agrochemical', 'volatile_oil', 'antibacterial_agent', 'insecticide'
    ]
    
    # Build Doc2Vec model
    model = build_doc2vec_model(corpus, maccs_list)
    
    # Generate compound vectors
    compound_vec = add_vec(maccs_list, model)
    X_vec_maccs = np.array([compound_vec[i] for i in range(len(df_maccs))])
    
    # Create index mapping
    index_mapping = create_index_mapping(len(df), invalid_indices)
    
    # Evaluate model performance
    results = evaluate_with_maccs(df, df_maccs, X_vec_maccs, categories, index_mapping)
    
if __name__ == "__main__":
    main()
