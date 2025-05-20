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
from shap_ecfp import generate_morgan_fingerprints, create_lightgbm_classifier

class _XOR_Tabular(shap.maskers.Independent):
    def __init__(self, data: np.ndarray, max_samples: int = 100):
        """
        XOR masker for tabular data
        
        Args:
            data: Original data array
            max_samples: Maximum number of samples for evaluation
        """
        super().__init__(data, max_samples=max_samples)

    def __call__(self, mask: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray]:
        """
        Apply XOR mask to input data
        
        Args:
            mask: Binary mask array
            x: Input data array
            
        Returns:
            Tuple containing the masked data array
        """
        mask = self._standardize_mask(mask, x)
        if np.issubdtype(mask.dtype, np.integer):
            super().__call__(mask, x)
        else:
            self._masked_data[:] = 1 - np.logical_xor(mask, x).reshape(1, len(x))
            self._last_mask[:] = mask
            return (self._masked_data, )

def _make_embed_pipeline(embeds: np.ndarray, model: BaseEstimator) -> Pipeline:
    """
    Create a pipeline with embedding transformation and classifier
    
    Args:
        embeds: Embedding vectors array
        model: Sklearn estimator model
        
    Returns:
        Sklearn pipeline with embedding transformer and classifier
    """
    def embedding_transform(X: np.ndarray, embeds: np.ndarray = embeds) -> np.ndarray:
        """
        Transform input array using embeddings
        
        Args:
            X: Input array to transform
            embeds: Embedding vectors
            
        Returns:
            Transformed array with embeddings
        """
        X_embeds = np.apply_along_axis(lambda x: embeds[np.where(x == 1)[0]].sum(axis=0), axis=1, arr=X)
        return X_embeds

    stacks = [('embedder', FunctionTransformer(embedding_transform)), ('classifier', model)]
    pipeline = Pipeline(stacks)
    return pipeline


def _xor_masker(mask: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Apply XOR mask to input data
    
    Args:
        mask: Binary mask array
        x: Input data array
        
    Returns:
        Masked data using XOR operation
    """
    return 1 - np.logical_xor(mask, x).reshape(1, -1)


def _normal_masker(mask: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Apply normal mask to input data
    
    Args:
        mask: Binary mask array
        x: Input data array
        
    Returns:
        Masked data using multiplication
    """
    return (x * mask).reshape(1, -1)


def shap_variables(embeds: np.ndarray, model: BaseEstimator, *, mask: Union[Callable, str] = 'xor') -> Tuple[Pipeline, Callable]:
    """
    Create a pipeline and masker for SHAP analysis
    
    Args:
        embeds: Embedding vectors array
        model: Sklearn estimator model
        mask: Masking method - 'xor', 'normal', or custom masking function
        
    Returns:
        Tuple containing (pipeline, masker function)
    """
    pipeline = _make_embed_pipeline(embeds, model)

    if mask == 'xor':
        masker = _xor_masker
    elif mask == 'normal':
        masker = _normal_masker
    else:
        masker = mask

    return pipeline, masker


def shap_additive_variables(embeds: np.ndarray, model: BaseEstimator, data: np.ndarray, *, mask: Union[Callable, str] = 'xor', max_samples: int = 100) -> Tuple[Pipeline, shap.maskers.Independent]:
    """
    Create a pipeline and masker for additive SHAP analysis
    
    Args:
        embeds: Embedding vectors array
        model: Sklearn estimator model
        data: Original data array
        mask: Masking method - 'xor', 'normal', or custom masking class
        max_samples: Maximum number of samples for evaluation
        
    Returns:
        Tuple containing (pipeline, masker class instance)
    """
    pipeline = _make_embed_pipeline(embeds, model)

    if mask == 'xor':
        masker = _XOR_Tabular(data=data, max_samples=max_samples)
    # elif mask == 'normal':  # TODO
        # masker = _normal_masker
    else:
        masker = mask(data=data, max_samples=max_samples)

    return pipeline, masker


def shap_visualize(shap_values: shap.Explanation,
                   show_option: bool = False,
                   *,
                   kinds: List[str] = ['bar', 'heatmap', 'beeswarm', 'violin'],
                   plot_kwg: Dict[str, Optional[Dict[str, Any]]] = {'bar': None, 'heatmap': None, 'beeswarm': None, 'violin': None}) -> None:
    """
    Visualize SHAP values using different plot types
    
    Args:
        shap_values: SHAP explanation object
        show_option: Whether to display plots immediately
        kinds: List of plot types to generate
        plot_kwg: Dictionary of keyword arguments for each plot type
        
    Returns:
        None
    """
    file_path = os.getcwd() + '/shap_visual'
    os.makedirs(file_path, exist_ok=True)
    os.chdir(file_path)
    
    for kind in kinds:
        if kind == 'bar':
            kwargs = plot_kwg['bar'] or {}
            shap.plots.bar(shap_values, show=show_option, **kwargs)

        if kind == 'heatmap':
            kwargs = plot_kwg['heatmap'] or {}
            shap.plots.heatmap(shap_values, show=show_option, **kwargs)

        if kind == 'beeswarm':
            kwargs = plot_kwg['beeswarm'] or {}
            shap.plots.beeswarm(shap_values, show=show_option, **kwargs)

        if kind == 'violin':
            kwargs = plot_kwg['violin'] or {}
            shap.plots.violin(shap_values, show=show_option, **kwargs)

def main(input_path: str, purpose: str, model_path: str, params: Dict[str, Any], max_evals: int, output_path: str) -> None:
    """
    Main function to run the SHAP analysis pipeline
    
    Args:
        input_path: Path to the pickled DataFrame with molecule data
        purpose: Target biological role to analyze
        model_path: Path to the Doc2Vec model
        params: Dictionary of parameters for LightGBM classifier configuration
        max_evals: Maximum evaluations for SHAP analysis
        output_path: Path to save the SHAP values output
        
    Returns:
        None
    """
    # Load dataset
    with open(input_path, "rb") as f:
        df = pickle.load(f)
    
    # Load model
    model = Doc2Vec.load(model_path)
    
    # Create classifier
    lightgbm = create_lightgbm_classifier(params)
    pipeline, masker = shap_variables(model.dv.vectors, lightgbm)
    
    # Generate target variable
    y = np.array([1 if i == purpose else 0 for i in df[purpose]])
    
    # Generate molecular fingerprints
    fingerprint = generate_morgan_fingerprints(df, 3, 4096)
    
    # Train model
    pipeline.fit(fingerprint, y)
    
    # Create explainer
    explainer = shap.Explainer(lambda x: pipeline.predict_proba(x)[:, 1], masker=masker)
    
    # Calculate SHAP values
    value = explainer(fingerprint, max_evals)
    
    # Save SHAP values
    with open(output_path, 'wb') as f:
        pickle.dump(value, f)


if __name__ == "__main__":
    # Example usage - replace with your actual file paths
    input_path = "10genre_dataset.pkl"
    model_path = "fpdoc2vec.model"
    purpose = "antioxidant"
    output_path = "shap_values.pkl"
    # Example params - replace with your actual params
    gbm_params: Dict[str, Any] = {
        "boosting_type": "dart", 
        "n_estimators": 444, 
        "learning_rate": 0.07284380689492893, 
        "max_depth": 6, 
        "num_leaves": 41, 
        "min_child_samples": 21, 
        "class_weight": "balanced", 
        "reg_alpha": 1.4922729949843299, 
        "reg_lambda": 2.8809246344115778, 
        "colsample_bytree": 0.5789063337359206, 
        "subsample": 0.5230422589468584, 
        "subsample_freq": 2, 
        "drop_rate": 0.1675163179873052, 
        "skip_drop": 0.49103811434109507, 
        "objective": 'binary', 
        "random_state": 50
    }
    main(input_path, purpose, model_path, gbm_params, max_evals = 500000, output_path)
