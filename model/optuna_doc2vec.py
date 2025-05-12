import numpy as np
import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import optuna

def add_vectors(fingerprint_df, model):
    """Combine document vectors based on fingerprints"""
    compound_vec = []
    for i in fingerprint_df:
        fingerprint_vec = 0
        for j in i:
            fingerprint_vec += model.dv.vectors[j]
        compound_vec.append(fingerprint_vec)
    return compound_vec

def process_single_category(args):
    """Process a single category for cross-validation"""
    category, train_df, params = args

    train_category_scores = []
    test_category_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    y = np.array([1 if i == category else 0 for i in train_df[category]])
    finger_list = list(train_df["fp_3_4096"])

    for train_idx, test_idx in skf.split(range(len(train_df)), y):
        # Prepare training data
        cm_train_df = train_df.iloc[train_idx]
        train_finger_list = list(cm_train_df["fp_3_4096"])

        # Prepare document corpus
        corpus = [sum(doc, []) for doc in cm_train_df["description_remove_stop_words"]]
        tagged_documents = [TaggedDocument(words=corpus, tags=train_finger_list[i]) for i, corpus in enumerate(corpus)]

        # Train Doc2Vec model
        model = Doc2Vec(tagged_documents, vector_size=100, min_count=0,
                      window=params['window'],
                      min_alpha=params["min_alpha"],
                      sample=params['sample'],
                      epochs=params['epochs'],
                      negative=params['negative'],
                      ns_exponent=params['ns_exponent'],
                      workers=1, seed=0)

        # Create compound vectors
        compound_vec = add_vectors(finger_list, model)
        X_vec = StandardScaler().fit_transform(
            np.array([compound_vec[i] for i in range(len(train_df))])
        )

        # Split data
        X_train_vec, X_test_vec = X_vec[train_idx], X_vec[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train and evaluate classifier
        logreg = LogisticRegression(n_jobs=1)
        logreg.fit(X_train_vec, y_train)

        # Calculate scores
        y_train_pred = logreg.predict(X_train_vec)
        y_test_pred = logreg.predict(X_test_vec)

        train_category_scores.append(f1_score(y_train, y_train_pred))
        test_category_scores.append(f1_score(y_test, y_test_pred))

    return np.mean(train_category_scores), np.mean(test_category_scores)

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    # Define parameter search space
    params = {
        'window': trial.suggest_int('window', 3, 10),
        'min_alpha': trial.suggest_float('min_alpha', 0.000001, 0.025),
        'sample': trial.suggest_float('sample', 0, 0.00001),
        'epochs': trial.suggest_int('epochs', 30, 1000),
        'negative': trial.suggest_int('negative', 1, 20),
        'ns_exponent': trial.suggest_float('ns_exponent', 0, 1)
    }

    # Process all categories in parallel
    with Pool(processes=cpu_count() - 1) as p:
        args = [(category, train_df, params) for category in categories]
        chunk_size = max(len(categories) // (cpu_count() - 1), 1)
        scores = p.map(process_single_category, args, chunksize=chunk_size)

    # Unpack scores
    train_scores = [score[0] for score in scores]
    test_scores = [score[1] for score in scores]

    # Calculate mean scores
    mean_train_score = np.mean(train_scores)
    mean_test_score = np.mean(test_scores)

    # Save training score as attribute
    trial.set_user_attr('train_f1', mean_train_score)

    return mean_test_score

class ProgressCallback:
    """Callback to display optimization progress"""
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
    """Main optimization function"""
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, callbacks=[ProgressCallback(n_trials)])

    print("\nBest trial:")
    print(f"  Test F1: {study.best_value:.4f}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    return study.best_trial.params

if __name__ == "__main__":
    # Load training data
    with open("data/train_df.pkl", "rb") as f:
        train_df = pickle.load(f)

    # Define categories
    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye',
        'toxin', 'flavouring_agent', 'agrochemical', 'volatile_oil',
        'antibacterial_agent', 'insecticide'
    ]

    # Run optimization
    best_params = optimize_doc2vec(150)
