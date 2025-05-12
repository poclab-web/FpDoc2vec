def addvec(fingerprint_df, model):
  compound_vec = []
  for i in fingerprint_df:
    fingerprint_vec = 0
    for j in i:
      fingerprint_vec += model.dv.vectors[j]
    compound_vec.append(fingerprint_vec)
  return compound_vec


def objective(trial):
    # パラメータの探索範囲を定義
    params = {
        'window': trial.suggest_int('window', 3, 10),
        'min_alpha': trial.suggest_float('min_alpha', 0.000001, 0.025),
        # 'min_count': trial.suggest_int('min_count', 0, 10),
        'sample': trial.suggest_float('sample', 0, 0.00001),
        'epochs': trial.suggest_int('epochs', 30, 1000),
        'negative': trial.suggest_int('negative', 1, 20),
        'ns_exponent': trial.suggest_float('ns_exponent', 0, 1)}

    with Pool(processes=cpu_count() - 1) as p:
        args = [(category, train_df, params) for category in categories]
        chunk_size = max(len(categories) // (cpu_count() - 1), 1)
        scores = p.map(process_single_category, args, chunksize=chunk_size)

    # アンパックして訓練データとテストデータのスコアを分離
    train_scores = [score[0] for score in scores]
    test_scores = [score[1] for score in scores]

    # 平均スコアを計算
    mean_train_score = np.mean(train_scores)
    mean_test_score = np.mean(test_scores)

    # スコアを trial に保存
    trial.set_user_attr('train_f1', mean_train_score)

    return mean_test_score


def process_single_category(args):
    category, train_df, params = args

    train_category_scores = []
    test_category_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    y = np.array([1 if i == category else 0 for i in train_df[category]])
    finger_list = list(train_df["fp_3_4096"])

    for train_idx, test_idx in skf.split(range(len(train_df)), y):
        cm_train_df = train_df.iloc[train_idx]
        train_finger_list = list(cm_train_df["fp_3_4096"])

        corpus = [sum(doc, []) for doc in cm_train_df["description_remove_stop_words"]]
        tagged_documents = [TaggedDocument(words=corpus, tags=train_finger_list[i]) for i, corpus in enumerate(corpus)]

        model = Doc2Vec(tagged_documents, vector_size=100, min_count=0,
                        window=params['window'],
                        min_alpha=params["min_alpha"],
                        sample=params['sample'],
                        epochs=params['epochs'],
                        negative=params['negative'],
                        ns_exponent=params['ns_exponent'],
                        workers=1, seed=0)

        compound_vec = addvec(finger_list, model)
        X_vec = StandardScaler().fit_transform(
            np.array([compound_vec[i] for i in range(len(train_df))])
        )

        X_train_vec, X_test_vec = X_vec[train_idx], X_vec[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        logreg = LogisticRegression(n_jobs=1)
        logreg.fit(X_train_vec, y_train)

        # 訓練データとテストデータの両方のスコアを計算
        y_train_pred = logreg.predict(X_train_vec)
        y_test_pred = logreg.predict(X_test_vec)

        train_category_scores.append(f1_score(y_train, y_train_pred))
        test_category_scores.append(f1_score(y_test, y_test_pred))

    return np.mean(train_category_scores), np.mean(test_category_scores)


class ProgressCallback:
    def __init__(self, n_trials):
        self.n_trials = n_trials
        self.pbar = tqdm(total=n_trials, desc="Optimization Progress")

    def __call__(self, study, trial):
        self.pbar.update(1)
        self.pbar.set_postfix({
            'Best Test F1': f"{study.best_value:.4f}",
            'Trial Test F1': f"{trial.value:.4f}",
            'Trial Train F1': f"{trial.user_attrs['train_f1']:.4f}"
        })


def optimize_doc2vec(n_trials):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, callbacks=[ProgressCallback(n_trials)])

    print("\nBest trial:")
    print(f"  Test F1: {study.best_value:.4f}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # 訓練データとテストデータの推移を可視化
    plt.figure(figsize=(12, 6))

    # テストデータの推移
    test_scores = [trial.value for trial in study.trials]
    plt.plot(test_scores, label='Test F1', color='blue')

    # 訓練データの推移
    train_scores = [trial.user_attrs['train_f1'] for trial in study.trials]
    plt.plot(train_scores, label='Train F1', color='red')

    plt.xlabel('Trial')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig("result/parameter/optuna_optimization_history.png")
    plt.show()

    return study.best_trial.params


if __name__ == "__main__":

    train_df = None
    categories = None

    # データの読み込み
    with open("chemdata/train_df.pkl", "rb") as f:
        train_df = pickle.load(f)

    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye',
        'toxin', 'flavouring_agent', 'agrochemical', 'volatile_oil',
        'antibacterial_agent', 'insecticide'
    ]

    best_params = optimize_doc2vec(150)
