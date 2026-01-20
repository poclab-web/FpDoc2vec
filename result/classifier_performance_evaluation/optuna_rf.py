import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
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

warnings.filterwarnings('ignore', category=optuna._experimental.ExperimentalWarning)


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

def process_single_fold_rf(args):
    category, train_df, params, train_idx, test_idx, fold_idx, model_dir = args
    try:
        # 保存されたDoc2Vecモデルを読み込み
        model = load_doc2vec_model(category, fold_idx, model_dir)

        y = np.array([1 if i == category else 0 for i in train_df[category]])
        finger_list = list(train_df["fp_3_4096"])

        # Doc2Vecでベクトル化
        compound_vec = addvec(finger_list, model)
        X_vec = np.array(compound_vec)

        X_train_vec, X_test_vec = X_vec[train_idx], X_vec[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        
        rf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            criterion=params["criterion"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            min_weight_fraction_leaf=params["min_weight_fraction_leaf"],
            max_features=params["max_features"],
            max_samples=params["max_samples"],
            bootstrap=params["bootstrap"],
            class_weight=params["class_weight"],
            random_state=0,
            n_jobs=1,
            verbose=0
        )
        rf.fit(X_train_vec, y_train)

        y_train_pred = rf.predict(X_train_vec)
        y_test_pred = rf.predict(X_test_vec)

        return category, fold_idx, f1_score(y_train, y_train_pred), f1_score(y_test, y_test_pred)

    except ValueError as e:
        
        print(f"Error in {category} fold {fold_idx}: {e}")
        return category, fold_idx, 0.0, 0.0


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'criterion': trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.3),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight',
                                                  ['balanced', 'balanced_subsample', None]),
    }

    # max_featuresの処理を修正
    max_features_type = trial.suggest_categorical('max_features_type', ['sqrt', 'log2', 'ratio', 'all'])
    
    if max_features_type == 'sqrt':
        params['max_features'] = 'sqrt'
    elif max_features_type == 'log2':
        params['max_features'] = 'log2'
    elif max_features_type == 'ratio':
        params['max_features'] = trial.suggest_float('max_features_ratio', 0.1, 1.0)
    else:  # 'all'
        params['max_features'] = None
    
    # bootstrap=Trueの場合のみmax_samplesを設定
    if params['bootstrap']:
        params['max_samples'] = trial.suggest_float('max_samples', 0.5, 1.0)
    else:
        params['max_samples'] = None

    all_tasks = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    for category in categories:
        y = np.array([1 if i == category else 0 for i in train_df[category]])
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(range(len(train_df)), y)):
            all_tasks.append((category, train_df, params,train_idx, test_idx, fold_idx, model_dir))

    total_cores = cpu_count()

    with Pool(processes=total_cores) as p:
        results = p.map(process_single_fold_rf, all_tasks)

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
        self.pbar.update(1)
        self.pbar.set_postfix({
            'Best Test F1': f"{study.best_value:.4f}",
            'Trial Test F1': f"{trial.value:.4f}",
            'Trial Train F1': f"{trial.user_attrs['train_f1']:.4f}"
        })
# 最適化の実行関数
def optimize_rf(n_trials):
    module = optunahub.load_module(package="samplers/auto_sampler")
    study = optuna.create_study(direction='maximize', sampler=module.AutoSampler())
    study.optimize(objective, n_trials=n_trials, callbacks=[ProgressCallback(n_trials)])

    print("\nBest trial:")
    print(f"  Test F1: {study.best_value:.4f}")
    print(f"  Train F1: {study.best_trial.user_attrs['train_f1']:.4f}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    plt.figure(figsize=(12, 6))

    test_scores = [trial.value for trial in study.trials]
    plt.plot(test_scores, label='Test F1', color='blue')

    train_scores = [trial.user_attrs['train_f1'] for trial in study.trials]
    plt.plot(train_scores, label='Training F1', color='red')

    plt.xlabel('Trial')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig("parameter/history_rf.png")
    plt.show()

    return study.best_trial.params


if __name__ == "__main__":
    
    with open("train_df2.pkl", "rb") as f:
        train_df = pickle.load(f)

    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye',
        'toxin', 'flavouring_agent', 'agrochemical', 'volatile_oil',
        'antibacterial_agent', 'insecticide'
    ]
    model_dir = "doc2vec_models"

    best_params = optimize_rf(500)

    with open("parameter/best_rf_params.pkl", "wb") as f:
        pickle.dump(best_params, f)