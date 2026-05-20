import sklearn
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import os
import shap
import warnings

warnings.filterwarnings('ignore', message='X does not have valid feature names')


def calculate_shap_values(model: sklearn.base.BaseEstimator, features: np.ndarray, max_evals: int = None) -> shap.Explanation:
    """Compute SHAP explanation values for a given model and input features."""
    explainer = shap.Explainer(model)
    if max_evals:
        return explainer(features, max_evals=max_evals)
    return explainer(features)


class _XOR_Tabular(shap.maskers.Independent):
    def __init__(self, data: np.ndarray, max_samples: int = 100):
        """Initialize the XOR-based tabular masker with background data."""
        super().__init__(data, max_samples=max_samples)

    def __call__(self, mask: np.ndarray, x: np.ndarray):
        """Apply XOR-based masking to generate masked input data for SHAP explanation."""
        mask = self._standardize_mask(mask, x)
        if np.issubdtype(mask.dtype, np.integer):
            super().__call__(mask, x)
        else:
            self._masked_data[:] = 1 - np.logical_xor(mask, x).reshape(1, len(x))
            self._last_mask[:] = mask
            return (self._masked_data, )


def _make_embed_pipeline(embeds: np.ndarray, model: sklearn.base.BaseEstimator) -> Pipeline:
    """Build a sklearn Pipeline that sums Doc2Vec embeddings for on-bits before classification."""
    def embedding_transform(X, embeds=embeds):
        return np.apply_along_axis(lambda x: embeds[np.where(x == 1)[0]].sum(axis=0), axis=1, arr=X)

    return Pipeline([('embedder', FunctionTransformer(embedding_transform)), ('classifier', model)])


def _xor_masker(mask: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return a masked array where bits are flipped using XOR between mask and input."""
    return 1 - np.logical_xor(mask, x).reshape(1, -1)


def _normal_masker(mask: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return a masked array by zeroing out input bits where the mask is inactive."""
    return (x * mask).reshape(1, -1)


def shap_variables(embeds: np.ndarray, model: sklearn.base.BaseEstimator, *, mask: Union[Callable, str] = 'xor') -> Tuple[Pipeline, Callable]:
    """Build an embedding pipeline and masker function for SHAP explanation of FpDoc2Vec features."""
    pipeline = _make_embed_pipeline(embeds, model)
    if mask == 'xor':
        masker = _xor_masker
    elif mask == 'normal':
        masker = _normal_masker
    else:
        masker = mask
    return pipeline, masker


def shap_additive_variables(embeds: np.ndarray, model: sklearn.base.BaseEstimator, data: np.ndarray, *, mask: Union[Callable, str] = 'xor', max_samples: int = 100) -> Tuple[Pipeline, Callable]:
    """Build an embedding pipeline and tabular masker for additive SHAP explanation using background data."""
    pipeline = _make_embed_pipeline(embeds, model)
    if mask == 'xor':
        masker = _XOR_Tabular(data=data, max_samples=max_samples)
    else:
        masker = mask(data=data, max_samples=max_samples)
    return pipeline, masker


def shap_visualize(shap_values: shap.Explanation, show_option: bool = False, *,
                   kinds: List[str] = ['bar', 'heatmap', 'beeswarm', 'violin'],
                   plot_kwg: Dict[str, Optional[Dict]] = {'bar': None, 'heatmap': None, 'beeswarm': None, 'violin': None}) -> None:
    """Save SHAP summary plots (bar, heatmap, beeswarm, violin) to a local directory."""
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
