import sklearn
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Callable
import numpy as np
from typing import Union
import os
import shap
import pickle
from gensim.models import Doc2Vec
import lightgbm as lgb
from rdkit.Chem import AllChem


class _XOR_Tabular(shap.maskers.Independent):
  def __init__(self, data: np.ndarray, max_samples: int =100):
    super().__init__(data, max_samples=max_samples)

  def __call__(self, mask, x):
    mask = self._standardize_mask(mask, x)
    if np.issubdtype(mask.dtype, np.integer):
      super().__call__(mask, x)
    else:
      self._masked_data[:] = 1 - np.logical_xor(mask, x).reshape(1, len(x))
      self._last_mask[:] = mask
      return (self._masked_data, )

def _make_embed_pipeline(embeds: np.ndarray, model: sklearn.base.BaseEstimator) -> Pipeline:
  def embedding_transform(X, embeds=embeds):
    X_embeds = np.apply_along_axis(lambda x: embeds[np.where(x == 1)[0]].sum(axis=0), axis=1, arr=X)
    return X_embeds

  stacks = [('embedder', FunctionTransformer(embedding_transform)), ('classifier', model)]
  pipeline = Pipeline(stacks)
  return pipeline

def _xor_masker(mask, x):
  return 1 - np.logical_xor(mask, x).reshape(1, -1)

def _normal_masker(mask, x):
  return (x * mask).reshape(1, -1)

def shap_variables(embeds: np.ndarray, model: sklearn.base.BaseEstimator, *, mask: Union[Callable, str]  = 'xor') -> Tuple[Pipeline, Callable]:
  '''
  in
  embeds: (arr), embeddings
  model: (sklearn model), model of sklearn to predict.
  mask: (Callable, str), if 'xor', use xor mask. If 'normal', use normal mask. If Callable, use custom mask.

  out
  pipeline: (sklearn model), wrapping model of sklearn to predict.
  masker: (Callable), mask function.
  '''
  pipeline = _make_embed_pipeline(embeds, model)

  if mask == 'xor':
    masker = _xor_masker
  elif mask == 'normal':
    masker = _normal_masker
  else:
    masker = mask

  return pipeline, masker

def shap_additive_variables(embeds: np.ndarray, model: sklearn.base.BaseEstimator, data: np.ndarray, *, mask: Union[Callable, str]  = 'xor', max_samples: int = 100) -> Tuple[Pipeline, Callable]:
  '''
  in
  embeds: (arr), embeddings
  model: (sklearn model), model of sklearn to predict.
  data: (arr), original data.
  mask: (Callable, str), if 'xor', use xor mask. If 'normal', use normal mask. If Callable, use custom mask.
  max_samples: (int), max samples for evaluation.

  out
  pipeline: (sklearn model), wrapping model of sklearn to predict.
  masker: (Independent), mask class.
  '''
  pipeline = _make_embed_pipeline(embeds, model)

  if mask == 'xor':
    masker = _XOR_Tabular(data=data, max_samples=max_samples)
  # elif mask == 'normal':  # TODO
    # masker = _normal_masker
  else:
    masker = mask(data=data, max_samples=max_samples)

  return pipeline, masker

def shap_visualize(shap_values: shap.Explanation,
                   show_option=False,
                   *,
                   kinds=['bar',
                          'heatmap',
                          'beeswarm',
                          'violin'],
                   plot_kwg={'bar':None, 'heatmap':None, 'beeswarm':None, 'violin':None, }) -> None:

  file_path = os.getcwd() + '/shap_visual'
  os.makedirs(file_path, exist_ok=True)
  os.chdir(file_path)
  for kind in kinds:
    if kind == 'bar':
      shap.plots.bar(shap_values, show=show_option, **plot_kwg['bar'])

    if kind == 'heatmap':
      shap.plots.heatmap(shap_values, show=show_option, **plot_kwg['heatmap'])

    if kind == 'beeswarm':
      shap.plots.beeswarm(shap_values, show=show_option, **plot_kwg['beeswarm'])

    if kind == 'violin':
      shap.plots.violin(shap_values, show=show_option, **plot_kwg['violin'])

def finger(df):
  fingerprints = []
  for i, mol in enumerate(df["ROMol"]):
    try:
      fingerprint = [j for j in AllChem.GetMorganFingerprintAsBitVect(mol, 3, 4096)]
      fingerprints.append(fingerprint)
    except:
      print("Error", i)
      continue
  fingerprints = np.array(fingerprints)
  return fingerprints

def create_lightgbm_classifier():
    """
    Create and configure LightGBM classifier with optimized parameters
    """
    return lgb.LGBMClassifier(
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

if __name__ == "__main__":

  # data loding
  with open("../../data/10genre_dataset.pkl", "rb") as f:
    df = pickle.load(f)
  model = Doc2Vec.load("../../model/fpdoc2vec4096.model")

  lightgbm = create_lightgbm_classifier()
  pipeline, masker = shap_variables(model.dv.vectors, lightgbm)

  # Specify the target biological role here
  # This example uses "antioxidant". To analyze other roles, replace "antioxidant" with:
  # "anti-inflammatory agent", "allergen", "dye", "toxin", "flavouring agent", 
  # "agrochemical", "volatile oil", "antibacterial agent", or "insecticide"
 
  y = np.array([1 if i == 'antioxidant' else 0 for i in df['antioxidant']])
  fingerprint = finger(df)
  pipeline.fit(fingerprint, y) 
  explainer = shap.Explainer(lambda x: pipeline.predict_proba(x)[:, 1], masker=masker)
  value = explainer(a, max_evals=500000)
  with open('shap_value/antioxidant_xor500000.pkl', 'wb') as f:
    pickle.dump(value, f)
