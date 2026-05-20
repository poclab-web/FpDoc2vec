from typing import Any, List
import pandas as pd
from gensim.models.doc2vec import TaggedDocument

def build_tagged_corpus(df: pd.DataFrame, tag_list: List[Any], description_column: str) -> List[TaggedDocument]:
    """Convert a DataFrame of preprocessed text into a tagged corpus."""
    tagged = []
    for i, doc in enumerate(df[description_column]):
        words = sum(doc, [])  
        tagged.append(TaggedDocument(words=words, tags=[tag_list[i]]))
    return tagged
