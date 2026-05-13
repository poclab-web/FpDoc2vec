from typing import Any, Dict, List
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def build_doc2vec_model(
    corpus: List[List[str]],
    tag_list: List[List[int]],
    params: Dict[str, Any],
) -> Doc2Vec:
    """Train a Doc2Vec model using fingerprint bit indices as document tags.

    Args:
        corpus: Tokenized description text for each compound
        tag_list: Fingerprint on-bit indices for each compound, used as document tags
        params: Doc2Vec hyperparameters (e.g. vector_size, epochs, window, ...)

    Returns:
        Trained Doc2Vec model
    """
    tagged_documents = [
        TaggedDocument(words=corpus, tags=tag_list[i]) 
        for i, corpus in enumerate(corpus)
    ]
    return Doc2Vec(tagged_documents, **params)

def fingerprints_to_vectors(on_bits_list: List[List[int]], model: Doc2Vec) -> np.ndarray:
    """Create compound vectors by averaging Doc2Vec vectors at each on-bit position.

    Each compound vector is the element-wise mean of the document vectors
    indexed by its fingerprint's on-bit positions. Compounds with no on-bits
    are represented as zero vectors.

    Args:
        on_bits_list: Fingerprint on-bit indices for each compound
        model: Trained Doc2Vec model

    Returns:
        2D numpy array of shape (n_compounds, vector_size)
    """
    vectors = []
    for bits in on_bits_list:
        if len(bits) == 0:
            vectors.append(np.zeros(model.vector_size))
        else:
            vec = np.sum([model.dv.vectors[b] for b in bits], axis=0)
            vectors.append(vec)
    return np.array(vectors)
