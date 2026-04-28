import pickle
import warnings
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import optuna
import optunahub
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def compute_compound_vectors(fingerprint_df, model):
    """Aggregate Doc2Vec tag vectors for each compound's fingerprint list."""
    compound_vectors = []
    for fingerprints in fingerprint_df:
        vec = sum(model.dv.vectors[tag] for tag in fingerprints)
        compound_vectors.append(vec)
    return compound_vectors


def process_single_fold(args):
    category, train_df, params, train_idx, test_idx, fold_idx = args

    y = np.array([1 if label == category else 0 for label in train_df[category]])
    finger_list = list(train_df["fp_3_4096"])

    fold_train_df = train_df.iloc[train_idx]
    fold_finger_list = list(fold_train_df["fp_3_4096"])

    descriptions = fold_train_df["description_gensim"].tolist()
    tagged_documents = [
        TaggedDocument(words=desc, tags=fold_finger_list[i])
        for i, desc in enumerate(descriptions)
    ]

    model = Doc2Vec(
        tagged_documents,
        dm=params["dm"],
        vector_size=params["vector_size"],
        min_count=params["min_count"],
        window=params["window"],
        alpha=params["alpha"],
        sample=params["sample"],
        epochs=params["epochs"],
        negative=params["negative"],
        workers=1,
        seed=0,
    )

    X = np.array(compute_compound_vectors(finger_list, model))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)

    train_f1 = f1_score(y_train, clf.predict(X_train))
    test_f1 = f1_score(y_test, clf.predict(X_test))

    return category, fold_idx, train_f1, test_f1


def make_objective(train_df, categories):
    """Return an Optuna objective that closes over training data and categories."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    def objective(trial):
        vector_size = trial.suggest_int("vector_size", 50, 150, step=10)
        dm = trial.suggest_categorical("dm", [0, 1])
        if dm == 0:  # DBOW
            window = trial.suggest_int("window", 6, 20)
            epochs = trial.suggest_int("epochs", 50, 150, step=5)
        else:  # DM
            window = trial.suggest_int("window", 3, 10)
            epochs = trial.suggest_int("epochs", 100, 1000, step=10)

        params = {
            "vector_size": vector_size,
            "dm": dm,
            "window": window,
            "min_count": 0,
            "alpha": trial.suggest_float("alpha", 0.01, 0.05, log=True),
            "sample": trial.suggest_float("sample", 1e-6, 1e-4, log=True),
            "epochs": epochs,
            "negative": trial.suggest_int("negative", 5, 20),
        }

        all_tasks = []
        for category in categories:
            y = np.array([1 if label == category else 0 for label in train_df[category]])
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(range(len(train_df)), y)):
                all_tasks.append((category, train_df, params, train_idx, test_idx, fold_idx))

        with Pool(processes=cpu_count()) as pool:
            results = pool.map(process_single_fold, all_tasks)

        category_results = {cat: {"train": [], "test": []} for cat in categories}
        for category, _, train_score, test_score in results:
            category_results[category]["train"].append(train_score)
            category_results[category]["test"].append(test_score)

        train_mean = np.mean([np.mean(category_results[cat]["train"]) for cat in categories])
        test_mean = np.mean([np.mean(category_results[cat]["test"]) for cat in categories])

        trial.set_user_attr("train_f1", train_mean)
        return test_mean

    return objective


class ProgressCallback:
    def __init__(self, n_trials):
        self.pbar = tqdm(total=n_trials, desc="Optimization Progress")

    def __call__(self, study, trial):
        self.pbar.update(1)
        self.pbar.set_postfix({
            "Best Test F1": f"{study.best_value:.4f}",
            "Trial Test F1": f"{trial.value:.4f}",
            "Trial Train F1": f"{trial.user_attrs['train_f1']:.4f}",
        })

    def close(self):
        self.pbar.close()


def optimize_doc2vec(train_df, categories, n_trials, output_path):
    module = optunahub.load_module(package="samplers/auto_sampler")
    study = optuna.create_study(direction="maximize", sampler=module.AutoSampler())

    callback = ProgressCallback(n_trials)
    study.optimize(make_objective(train_df, categories), n_trials=n_trials, callbacks=[callback])
    callback.close()

    best = study.best_trial
    print("\nBest trial:")
    print(f"  Test F1:  {study.best_value:.4f}")
    print(f"  Train F1: {best.user_attrs['train_f1']:.4f}")
    print("  Params:")
    for key, value in best.params.items():
        print(f"    {key}: {value}")

    test_scores = [t.value for t in study.trials]
    train_scores = [t.user_attrs["train_f1"] for t in study.trials]

    plt.figure(figsize=(12, 6))
    plt.plot(test_scores, label="Test data", color="blue")
    plt.plot(train_scores, label="Training data", color="red")
    plt.xlabel("Trial")
    plt.ylabel("F1 score")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()

    return best.params


if __name__ == "__main__":
    CATEGORIES = [
        "antioxidant", "anti_inflammatory_agent", "allergen", "dye",
        "toxin", "flavouring_agent", "agrochemical", "volatile_oil",
        "antibacterial_agent", "insecticide",
    ]

    with open("train_df.pkl", "rb") as f:
        train_df = pickle.load(f)

    best_params = optimize_doc2vec(
        train_df=train_df,
        categories=CATEGORIES,
        n_trials=500,
        output_path="optuna_optimization_history_doc2vec.png",
    )
