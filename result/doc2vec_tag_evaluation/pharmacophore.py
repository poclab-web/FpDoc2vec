import pickle
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from ECFP2048bit.py import add_vec, build_doc2vec_model, create_lightgbm_classifier
from MACCSkeys import create_index_mapping

def process_pharmacophore_features(df, pharmacore_list):
    """
    Process pharmacophore features and identify invalid entries
    """
    df["pharmacore"] = pharmacore_list
    invalid_pharmacore_indices = []
    
    for idx, feature in enumerate(pharmacore_list):
        if feature is None:
            invalid_pharmacore_indices.append(idx)
            
    return invalid_pharmacore_indices

def evaluate_with_pharmacophore(df, df_pharm, X_vec_pharm, categories, index_mapping):
    """
    Evaluate model performance using pharmacophore features and Doc2Vec vectors
    """
    lightgbm_model = create_lightgbm_classifier()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    
    results = {}
    for category_idx, category in enumerate(categories):
        pharm_train_f1_list = []
        pharm_test_f1_list = []
        
        # Prepare target variables
        y = np.array([1 if i == category else 0 for i in df[category]])
        y_pharm = np.array([1 if i == category else 0 for i in df_pharm[category]])
        
        for train_idx, test_idx in skf.split(range(len(df)), y):
            # Convert indices to pharmacophore-compatible indices
            train_idx_pharm = [index_mapping[idx] for idx in train_idx if idx in index_mapping]
            test_idx_pharm = [index_mapping[idx] for idx in test_idx if idx in index_mapping]
            
            # Only proceed if we have enough samples in both train and test sets
            if len(train_idx_pharm) > 0 and len(test_idx_pharm) > 0:
                # Extract training and testing data
                X_train_pharm = X_vec_pharm[train_idx_pharm]
                X_test_pharm = X_vec_pharm[test_idx_pharm]
                y_train_pharm = y_pharm[train_idx_pharm]
                y_test_pharm = y_pharm[test_idx_pharm]
                
                # Train model and make predictions
                lightgbm_model.fit(X_train_pharm, y_train_pharm)
                y_pred_train_pharm = lightgbm_model.predict(X_train_pharm)
                y_pred_test_pharm = lightgbm_model.predict(X_test_pharm)
                
                # Calculate F1 score
                pharm_train_f1_list.append(f1_score(y_train_pharm, y_pred_train_pharm))
                pharm_test_f1_list.append(f1_score(y_test_pharm, y_pred_test_pharm))
        
        print(f"## {category} ##")
        print(f"Training Data: {np.mean(pharm_train_f1_list)}")
        print(f"Test Data: {np.mean(pharm_test_f1_list)}")
      
    return {
        'train_scores': maccs_train_f1_list,
        'test_scores': maccs_test_f1_list,
        'mean_train': np.mean(maccs_train_f1_list),
        'mean_test': np.mean(maccs_test_f1_list)
    }

def main():
    # Load dataset
    with open("../../data/10genre_dataset.pkl", "rb") as f:
        df = pickle.load(f)
    
    # Load pharmacophore features
    with open("../../model/10genre_3926pharmacore.pkl", "rb") as f:
        pharmacore_list = pickle.load(f)
    
    # Process pharmacophore features
    invalid_pharmacore_indices = process_pharmacophore_features(df, pharmacore_list)
    
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
