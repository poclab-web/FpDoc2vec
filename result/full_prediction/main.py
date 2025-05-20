def main(input_path,  X_vec):
    # Load dataset
    with open(input_path, "rb") as f:
        df = pickle.load(f)
    
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



# loading Doc2Vec model
    model = Doc2Vec.load(model_path)
    
    # Generate compound vectors
  finger_list = list(df["fp_3_4096"])
    compound_vec = add_vectors(finger_list, model)
    X_vec = np.array([compound_vec[i] for i in range(len(df))])

# loading Doc2Vec model
    model = Doc2Vec.load("../../model/namedoc2vec.model")
    
    # Generate compound vectors
    X_vec = np.array([model.dv.vectors[i] for i in range(len(df))])







# Load dataset
    with open("../../10genre_32descriptor.pkl", "rb") as f:
        df = pickle.load(f)
      # Generate fingerprint
    desc = np.array(fin(df))
