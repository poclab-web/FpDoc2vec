import pandas as pd
import numpy as np
from numpy.linalg import norm

from utils import CATEGORIES

def get_categories(df: pd.DataFrame, idx: int) -> str:
    
    found_categories = []
    for cat in CATEGORIES:
        if cat in df.columns and df.iat[idx, df.columns.get_loc(cat)] != 'No':
            found_categories.append(cat)
    return ', '.join(found_categories)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def calculate_similarities(df: pd.DataFrame, vectors: np.ndarray, target_compound_name: str) -> List[float]:
    try:
        
        target_indices = df[df["NAME"] == target_compound_name].index
        target_idx = target_indices[0]
        target_vector = vectors[target_idx]
        similarities = []
        for i, vector in enumerate(vectors):
            if i != target_idx:
                sim = cosine_similarity(vector, target_vector)
                similarities.append(sim)
            else:
                similarities.append(0)       
        return similarities
        
    except IndexError:
        raise ValueError(f"Compound '{target_compound_name}' not found in the dataset")

def similarity_output(df, compound_vec:np.ndarray, target_compound: str, n: int) -> None:
    
    # Calculate similarities to sucrose
    df[target_compound] = calculate_similarities(df, compound_vec, target_compound)
    
    # Get top n similar compounds
    sorted_df = df.sort_values(target_compound, ascending=False)
    
    # Return the top N compounds (excluding the target compound itself)
    top_similar = sorted_df.head(n)[['NAME', target_compound]]
    target_idx = df[df["NAME"] == target_compound].index[0]
    target_categories = get_categories(df, target_idx)
    print(f"Target compound '{target_compound}' categories: {target_categories}")

    # Display results in the same format as the previous code
    print(f"Top {n} most similar compounds to '{target_compound}':")
    for idx, row in top_similar.iterrows():
        compound_name = row['NAME']
        similarity_score = row[target_compound]
        categories = get_categories(df, idx)  
        print(f"  {compound_name}: {similarity_score:.4f} (Categories: {categories})")  
        display(df.at[idx, 'ROMol'])















def fin2(df, radius, fpSize):
    fingerprint_objects = []  # この行を追加
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)
    for i, mol in enumerate(df["ROMol"]):
        try:
            fp = fp_generator.GetFingerprint(mol)
            fingerprint_objects.append(fp)  # この行を追加

        except Exception as e:
            print(f"Error processing molecule {i}: {e}")
            continue
    return fingerprint_objects  

def calculate_tanimoto_similarities(df: pd.DataFrame, fingerprint_objects: List, target_compound: str) -> List[float]:
    try:
        # ターゲット化合物のインデックスを取得
        df["NAME"] = [df.iat[i, 0][0] for i in range(len(df))]
        target_indices = df[df["NAME"] == target_compound].index
        target_idx = target_indices[0]
        
        # ターゲットのフィンガープリント
        target_fp = fingerprint_objects[target_idx]
        
        # BulkTanimotoSimilarityで一括計算
        similarities = BulkTanimotoSimilarity(target_fp, fingerprint_objects)
        
        # 自分自身は0に設定
        similarities[target_idx] = 0.0
                
        return similarities
        
    except IndexError:
        raise ValueError(f"化合物 '{target_compound}' がデータセットに見つかりません")
    
    
def tanimoto_similarity_output(input_path: str, fingerprint_objects: List, target_compound: str, n: int = 10) -> None:
    """谷本類似度の結果を出力"""
    # Load dataset
    with open(input_path, "rb") as f:
        df = pickle.load(f)
    
    # Add compound names as a separate column
    df["NAME"] = [df.iat[i, 0][0] for i in range(len(df))]
    df[target_compound] = calculate_tanimoto_similarities(df, fingerprint_objects, target_compound)
    
    # Get top n similar compounds
    sorted_df = df.sort_values(target_compound, ascending=False)
    top_similar = sorted_df.head(n)[['NAME', target_compound]]
    target_idx = df[df["NAME"] == target_compound].index[0]
    target_categories = get_categories(df, target_idx)
    # Display results with 10th column
    print(f"Target compound '{target_compound}' categories: {target_categories}")
    print(f"Top {n} most similar compounds to '{target_compound}' (Tanimoto similarity):")
    for idx, row in top_similar.iterrows():
        compound_name = row['NAME']
        similarity_score = row[target_compound]
        categories = get_categories(df, idx)  # 追加: カテゴリー取得
        print(f"  {compound_name}: {similarity_score:.4f} (Categories: {categories})")
        display(df.iat[idx, 10])