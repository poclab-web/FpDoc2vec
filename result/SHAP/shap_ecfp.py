import numpy as np
import shap
import pickle
import lightgbm as lgb
from rdkit.Chem import AllChem
from sklearn.metrics import accuracy_score, roc_auc_score

def generate_fingerprints(df):
    """
    Generate Morgan fingerprints (ECFP) for molecules in the dataframe
    
    Args:
        df: DataFrame containing ROMol column with RDKit molecule objects
        
    Returns:
        numpy array of fingerprints
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

def train_lightgbm_model(fingerprints, target_values):
    """
    Train a LightGBM model with optimized hyperparameters
    
    Args:
        fingerprints: numpy array of molecular fingerprints
        target_values: numpy array of target values (0 or 1)
        
    Returns:
        trained LightGBM model
    """
    model = lgb.LGBMClassifier(
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
    
    model.fit(fingerprints, target_values)
    return model

def calculate_shap_values(model, features):
    """
    Calculate SHAP values for the trained model
    
    Args:
        model: trained LightGBM model
        features: feature matrix used for explanation
        
    Returns:
        SHAP values array
    """
    explainer = shap.TreeExplainer(
        model=model,
        feature_perturbation='tree_path_dependent',
        model_output='raw'
    )
    
    return explainer.shap_values(features)

def main():
    """Main function to run the SHAP analysis pipeline"""
    # Load dataset
    with open("../../data/10genre_dataset.pkl", "rb") as f:
        df = pickle.load(f)
    
    # Specify the target biological role here
    # This example uses "antioxidant". To analyze other roles, replace "antioxidant" with:
    # "anti-inflammatory agent", "allergen", "dye", "toxin", "flavouring agent", 
    # "agrochemical", "volatile oil", "antibacterial agent", or "insecticide"
    y = np.array([1 if i == 'antioxidant' else 0 for i in df['antioxidant']])
    
    # Generate molecular fingerprints
    fingerprints = generate_fingerprints(df)
    
    # Train the model
    model = train_lightgbm_model(fingerprints, y)
    
    # Calculate SHAP values
    shap_values = calculate_shap_values(model, fingerprints)
    
    # Save SHAP values
    with open('shap_value/antioxidant_ECFP.pkl', 'wb') as f:
        pickle.dump(shap_values, f)

if __name__ == "__main__":
    main()
