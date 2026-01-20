import numpy as np
import lightgbm as lgb
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
warnings.filterwarnings('ignore', message='Early stopping is not available in dart mode')


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

def process_single_fold_lightgbm(args):
    category, train_df, params, train_idx, test_idx, fold_idx, model_dir = args

    # 保存されたDoc2Vecモデルを読み込み
    model = load_doc2vec_model(category, fold_idx, model_dir)

    y = np.array([1 if i == category else 0 for i in train_df[category]])
    finger_list = list(train_df["fp_3_4096"])

    # Doc2Vecでベクトル化
    compound_vec = addvec(finger_list, model)
    X_vec = np.array(compound_vec)

    X_train_vec, X_test_vec = X_vec[train_idx], X_vec[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # LGBMClassifier初期化時にverbose設定を追加
    classifier_params = params.copy()
    classifier_params['verbose'] = -1  # 初期化時にverbose設定
    
    boosting_type = classifier_params.pop('boosting_type', 'gbdt')
    if boosting_type == 'dart':
        # dartモードの場合はearly stoppingなし
        callbacks = []
        lightgbm = lgb.LGBMClassifier(**classifier_params)
        lightgbm.fit(X_train_vec, y_train,
            eval_set=[(X_test_vec, y_test)],
            eval_metric='binary_logloss',
            callbacks=callbacks)
    else:
        # その他のモードは従来通りearly stopping使用
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        lightgbm = lgb.LGBMClassifier(**classifier_params)
        lightgbm.fit(X_train_vec, y_train,
            eval_set=[(X_test_vec, y_test)],
            eval_metric='binary_logloss',
            callbacks=callbacks)

    y_train_pred = lightgbm.predict(X_train_vec)
    y_test_pred = lightgbm.predict(X_test_vec)

    return category, fold_idx, f1_score(y_train, y_train_pred), f1_score(y_test, y_test_pred)

def objective(trial):

    boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])
    num_leaves = trial.suggest_int('num_leaves', 31, 255)  
    min_depth = max(3, int(np.log2(num_leaves)))
    max_depth = trial.suggest_int('max_depth', min_depth, 15)  

    # 基本パラメータを設定（警告を減らすために調整）
    params = {
        'boosting_type': boosting_type,
        'num_leaves': num_leaves,
        'max_depth': max_depth,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 750),
        "objective": 'binary',
        'metric': 'binary_logloss',
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.1),  
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-4,  1e-1, log=True), 
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),  
        # 適切な正則化範囲
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),  # L1正則化：適度な範囲
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),  # L2正則化：適度な範囲
        # 特徴量サンプリング（追加）
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'feature_fraction_bynode': trial.suggest_float('feature_fraction_bynode', 0.5, 1.0),
        "random_state": 0,
        # 警告を抑制
        'verbose': -1, 
        'force_col_wise': True,  
    }
    
    # ブースティングタイプ別のパラメータ設定
    if boosting_type == 'gbdt':
        params.update({
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # 元の範囲を維持
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # 元の範囲を維持
            'subsample_freq': trial.suggest_int('subsample_freq', 0, 10),  # 元の範囲を維持
        })
        if params['subsample'] == 1.0:
            params['subsample_freq'] = 0
        
    elif boosting_type == 'dart':
        params.update({
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 0, 10),
            'drop_rate': trial.suggest_float('drop_rate', 0.01, 0.3),  # 妥当な範囲に調整
            'max_drop': trial.suggest_int('max_drop', 20, 100),  # 妥当な範囲に調整
            'skip_drop': trial.suggest_float('skip_drop', 0.3, 0.7),  # 
            'uniform_drop': trial.suggest_categorical('uniform_drop', [True, False]),  # 追加
            'xgboost_dart_mode': trial.suggest_categorical('xgboost_dart_mode', [True, False]),  # 追加
        })
        if params['subsample'] == 1.0:
            params['subsample_freq'] = 0
        
    else:  # goss
        # GOSSの制約条件を考慮
        top_rate = trial.suggest_float('top_rate', 0.1, 0.5)
        # other_rateはtop_rateとの合計が1.0を超えないように制約
        max_other_rate = min(0.3, 0.95 - top_rate)
        other_rate = trial.suggest_float('other_rate', 0.01, max_other_rate)
        
        params.update({
            'top_rate': top_rate,
            'other_rate': other_rate,
            'subsample': 1.0,  # GOSSでは1.0固定
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        })
        # GOSSの場合、subsampleとsubsample_freqは使用しない
        params.pop('subsample_freq', None)
    # 不均衡データ対策の追加オプション
    if params['class_weight'] is None:
        params['is_unbalance'] = trial.suggest_categorical('is_unbalance', [True, False])
    else:
        params['is_unbalance'] = False


    # 全てのカテゴリ×フォールドの組み合わせを並列処理
    all_tasks = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    for category in categories:
        y = np.array([1 if i == category else 0 for i in train_df[category]])
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(range(len(train_df)), y)):
            all_tasks.append((category, train_df, params,
                                  train_idx, test_idx, fold_idx, model_dir))

    total_cores = cpu_count()

    with Pool(processes=total_cores) as p:
        results = p.map(process_single_fold_lightgbm, all_tasks)

    # 結果をカテゴリ別に整理
    category_results = {category: {'train': [], 'test': []} for category in categories}

    for category, fold_idx, train_score, test_score in results:
        category_results[category]['train'].append(train_score)
        category_results[category]['test'].append(test_score)

    # 各カテゴリの平均スコアを計算
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
def optimize_lightgbm(n_trials):
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
    plt.ylabel('F1 score')
    plt.legend()
    plt.grid(True)
    plt.savefig("parameter/history_lightgbm2.png")
    plt.show()

    return study.best_trial.params


if __name__ == "__main__":
    # データの読み込み
    with open("train_df2.pkl", "rb") as f:
        train_df = pickle.load(f)

    categories = [
        'antioxidant', 'anti_inflammatory_agent', 'allergen', 'dye',
        'toxin', 'flavouring_agent', 'agrochemical', 'volatile_oil',
        'antibacterial_agent', 'insecticide'
    ]
    model_dir = "doc2vec_models"

    # パラメータ最適化の実行
    best_params = optimize_lightgbm(3000)

    # 最適パラメータを保存
    with open("parameter/best_lightgbm_params2.pkl", "wb") as f:
        pickle.dump(best_params, f)