import pickle
import numpy as np
from rdkit.Chem import AllChem
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def generate_morgan_fingerprints(df):
    """
    Generate Morgan fingerprints (ECFP6) with radius 3 and 4096 bits for molecules in the dataframe.
    Returns a list of the bit positions that are set to 1 for each molecule.
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
    return [[j for j in range(4096) if i[j] == 1] for i in fingerprints]

def train_fingerprint_doc2vec_model(df, fingerprints):
    """
    Train a Doc2Vec model using document descriptions and molecular fingerprints as tags.
    """
    corpus = [sum(doc, []) for doc in df["description_remove_stop_words"]]
    tagged_documents = [TaggedDocument(words=corpus, tags=fingerprints[i]) 
                        for i, corpus in enumerate(corpus)]
    
    model = Doc2Vec(tagged_documents, 
                    vector_size=100, 
                    min_count=0,
                    window=10,
                    min_alpha=0.023491749982816976,
                    sample=7.343338709169564e-06,
                    epochs=859,
                    negative=2,
                    ns_exponent=0.8998927133390002,
                    workers=1, 
                    seed=100)
    
    return model

def main():
    # Load dataset
    with open("data/10genre_dataset.pkl", "rb") as f:
        df = pickle.load(f)
    
    # Generate fingerprints
    df["fp_3_4096"] = generate_morgan_fingerprints(df)
    finger_list = list(df["fp_3_4096"])
    
    # Define categories (not used in this code but kept for reference)
    categories = ['antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin', 
                 'flavouring_agent', 'agrochemical', 'volatile_oil', 
                 'antibacterial_agent', 'insecticide']
    
    # Train Doc2Vec model
    model = train_fingerprint_doc2vec_model(df, finger_list)
    
    # Save the model
    model.save("model/fpdoc2vec4096.model")

if __name__ == "__main__":
    main()
