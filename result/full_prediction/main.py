def main(df,  X_vec):
    
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
        y = np.array([1 if i == category else 0 for i in df[category]])
        results[category] = evaluate_category(category, X_vec, y, lightgbm_model)

def main():
input_path = "10genre_dataset.pkl"
# Load dataset
    with open(input_path, "rb") as f:
        df = pickle.load(f)
#fpdoc2vec
    model_path = "fpdoc2vec.model"
    fpvec = make_fp2vector(model_path, df)
# loading Doc2Vec model
    model = Doc2Vec.load(model_path)
main(df,  fpvec)




#namedoc2vec
    model_path = "namedoc2vec.model"
    namevec = make_name2vector(model_path, df)
# loading Doc2Vec model
    model = Doc2Vec.load(model_path)
main(df,  fpvec)


#ecfp
ecfp = np.array(generate_morgan_fingerprints(df))
main(df,  ecfp)

#descriptor
input_descriptor_path = "10genre_32descriptor.pkl"
desc = make_descriptor(input_descriptor_path)
main(df,  ecfp)
