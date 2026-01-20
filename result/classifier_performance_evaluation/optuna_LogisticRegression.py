import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import optuna
import matplotlib.pyplot as plt
import pickle
from gensim.models.doc2vec import Doc2Vec
import optunahub
from multiprocessing import Pool, cpu_count
import os
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=optuna._experimental.ExperimentalWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)


def addvec(fingerprint_df, model):
    compound_vec = []
    for i in fingerprint_df:
        fingerprint_vec = 0
        for j in i:
            fingerprint_vec += model.dv.vectors[j]
        compound_vec.append(fingerprint_vec)
    return compound_vec


def load_doc2vec_model(category, fold_idx, model_dir="doc2vec_models"):
    model_path = os.path.join(model_dir, f"doc2vec_{category}_fold{fold_idx}.model")
    return Doc2Vec.load(model_path)

def process_single_fold_lr(args):
    category, train_df, base_lr_params, train_idx, test_idx, fold_idx, model_dir = args

    model = load_doc2vec_model(category, fold_idx, model_dir)

    y = np.array([1 if i == category else 0 for i in train_df[category]])
    finger_list = list(train_df["fp_3_4096"])
    compound_vec = addvec(finger_list, model)
    X_vec = np.array(compound_vec)

    X_train_vec, X_test_vec = X_vec[train_idx], X_vec[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    lr = LogisticRegression(**base_lr_params)
    try:
        lr.fit(X_train_vec, y_train)
        y_train_pred = lr.predict(X_train_vec)
        y_test_pred = lr.predict(X_test_vec)

        return category, fold_idx, f1_score(y_train, y_train_pred), f1_score(y_test, y_test_pred)

    except ValueError as e:
        raise ValueError(f"Error in category {category}, fold {fold_idx}: {str(e)}") from e

def objective(trial):
    
    base_lr_params = {
        'C': trial.suggest_float('base_C', 0.05, 10.0, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', "elasticnet"]),
        'max_iter': trial.suggest_int('max_iter', 3000, 5000, step=100),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'tol': trial.suggest_float('tol', 1e-5, 1e-2, log=True),
        'random_state': 0
    }
    
    if base_lr_params["penalty"] == 'l1':
        base_lr_params['solver'] = trial.suggest_categorical('solver_l1', ['liblinear', 'saga'])
    elif base_lr_params["penalty"] == 'l2':  # l2
        base_lr_params['solver'] = trial.suggest_categorical('solver_l2', ['lbfgs', 'liblinear', 'saga', 'newton-cg'])
    else:  # elasticnet
        base_lr_params['solver'] = 'saga'  # ElasticNetはsagaのみ対応
        base_lr_params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.1, 0.9, step=0.1)

    all_tasks = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    for category in categories:
        y = np.array([1 if i == category else 0 for i in train_df[category]])
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(range(len(train_df)), y)):
            all_tasks.append((category, train_df, base_lr_params,
                              train_idx, test_idx, fold_idx, model_dir))

    total_cores = cpu_count()

    with Pool(processes=total_cores) as p:
        results = p.map(process_single_fold_lr, all_tasks)

   
    category_results = {category: {'train': [], 'test': []} for category in categories}

    for category, fold_idx, train_score, test_score in results:
        category_results[category]['train'].append(train_score)
        category_results[category]['test'].append(test_score)

    
    category_train_means = []
    category_test_means = []

    for category in categories:
        train_mean = np.mean(category_results[category]['train'])
        test_mean = np.mean(category_results[category]['test'])
        category_train_means.append(train_mean)
        category_test_means.append(test_mean)

    overall_train_mean = np.mean(category_train_means)
    overall_test_mean = np.mean(category_test_means)

    trial.set_user_attr('train_f1', overall_train_mean)
    return overall_test_mean


class ProgressCallback:
    def __init__(self, n_trials):
        self.n_trials = n_trials
        self.pbar = tqdm(total=n_trials, desc="Optimization Progress")

    def __call__(self, study, trial):
        self.last_value = trial.value
        self.pbar.update(1)
        self.pbar.set_postfix({
            'Best Test F1': f"{study.best_value:.4f}",
            'Trial Test F1': f"{self.last_value:.4f}",
            'Trial Train F1': f"{trial.user_attrs['train_f1']:.4f}"
        })

def optimize_lr(n_trials):
    module = optunahub.load_module(package="samplers/auto_sampler")
    study = optuna.create_study(direction='maximize', sampler=module.AutoSampler())
    study.optimize(objective, n_trials=n_trials, callbacks=[ProgressCallback(n_trials)])

    print("\nBest trial:")
    print(f"  Test F1: {study.best_value:.4f}")
    print(f"  Train F1: {study.best_trial.user_attrs['train_f1']:.4f}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # 訓練データとテストデータの推移を可視化
    plt.figure(figsize=(12, 6))

    # テストデータの推移
    test_scores = [trial.value for trial in study.trials]
    plt.plot(test_scores, label='Test data', color='blue')

    # 訓練データの推移
    train_scores = [trial.user_attrs['train_f1'] for trial in study.trials]
    plt.plot(train_scores, label='Training data', color='red')

    plt.xlabel('Trial')
    plt.ylabel('F1 score')
    plt.legend()
    plt.grid(True)
    plt.savefig("parameter/history_lr.png")
    plt.show()

    return study.best_trial.params


if __name__ == "__main__":
    # データの読み込み
    with open("data/train_df2.pkl", "rb") as f:
        train_df = pickle.load(f)

    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye',
        'toxin', 'flavouring_agent', 'agrochemical', 'volatile_oil',
        'antibacterial_agent', 'insecticide'
    ]
    model_dir = "doc2vec_models"

    # パラメータ最適化の実行
    best_params = optimize_lr(1000)

    # 最適パラメータを保存
    with open("parameter/best_lr_params.pkl", "wb") as f:
        pickle.dump(best_params, f)