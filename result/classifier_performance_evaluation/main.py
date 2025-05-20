from sklearn.linear_model import LogisticRegression

def main(traindf_path, testdf_path, model_path):
    """
    Main function to load data, prepare features, and evaluate models
    for different chemical categories
    """
    # Load data
    with open(traindf_path, "rb") as f:
        test_df = pickle.load(f)
    with open(testdf_path, "rb") as f:
        train_df = pickle.load(f)
    
    # Load model
    model = Doc2Vec.load(model_path)
    
    # Define categories
    categories = ['antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye', 'toxin', 
                  'flavouring_agent', 'agrochemical', 'volatile_oil', 
                  'antibacterial_agent', 'insecticide']
    
    # Prepare feature vectors
    train_finger_list = list(train_df["fp_3_4096"])
    test_finger_list = list(test_df["fp_3_4096"])
    
    train_compound_vec = addvec(train_finger_list, model)
    test_compound_vec = addvec(test_finger_list, model)
    
    X_train_vec = np.array([train_compound_vec[i] for i in range(len(train_df))])
    X_test_vec = np.array([test_compound_vec[i] for i in range(len(test_df))])
    
    # Evaluate each category
    train, test = [], []
    for category in categories:
        train_score, test_score = evaluate_category(category, 
X_vec: np.ndarray, 
                      y: np.ndarray, 
                      estimator_model
        train.append(train_score)
        test.append(test_score)
    
    # Optional: Calculate and print average scores
    print("\n## Average scores across all categories ##")
    print(f"Average Training F1: {np.mean(train):.4f}")
    print(f"Average Test F1: {np.mean(test):.4f}")

if __name__ == "__main__":
    #LR
    LogisticRegression
    main()
