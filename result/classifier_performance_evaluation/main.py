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
        train_score, test_score = evaluate_category(
            category, X_train_vec, X_test_vec, train_df, test_df
        )
        train.append(train_score)
        test.append(test_score)
    
    # Optional: Calculate and print average scores
    print("\n## Average scores across all categories ##")
    print(f"Average Training F1: {np.mean(train):.4f}")
    print(f"Average Test F1: {np.mean(test):.4f}")

if __name__ == "__main__":
    main()


(
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
