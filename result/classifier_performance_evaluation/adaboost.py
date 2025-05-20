
    dt = DecisionTreeClassifier(max_depth=2, class_weight="balanced")
    ada = AdaBoostClassifier(
        estimator=dt,
        n_estimators=442,
        learning_rate=0.07760379807069998,
        algorithm="SAMME.R",
        random_state=50
    )
