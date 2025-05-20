import pickle
from typing import List, Dict, Any
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def smiles_to_ngrams(smiles_list: List[str], n: int) -> List[List[str]]:
    """
    Convert SMILES strings to n-grams
    
    Args:
        smiles_list: List of SMILES strings
        n: Integer specifying the n-gram size
        
    Returns:
        List of lists containing n-grams for each SMILES string
    """
    ngrams_list = []
    for smiles in smiles_list:  # Fixed the variable name from smileslist to smiles_list
        if len(smiles) < n:
            ngrams_list.append([smiles])
        else:
            ngrams_list.append([smiles[i:i+n] for i in range(len(smiles) - n + 1)])
    return ngrams_list


def make_ngramlist(input_path: str, n: int = 3) -> List[List[int]]:
    """
    Generate binary n-gram vectors from SMILES strings in a pickle file
    
    Args:
        input_path: Path to the pickle file containing a DataFrame with a 'smiles' column
        n: Integer specifying the n-gram size (default=3)
        
    Returns:
        List of lists containing indices of present n-grams for each SMILES string
    """
    # Load dataset
    with open(input_path, "rb") as f:
        df = pickle.load(f)
      
    # Generate n-grams from SMILES   
    smiles_list = list(df["smiles"])
    ngrams_list = smiles_to_ngrams(smiles_list, n)

    # Convert n-grams to binary vectors
    vectorizer = CountVectorizer(binary=True, analyzer=lambda x: x)
    vec = vectorizer.fit_transform(ngrams_list)
  
    # Convert sparse vectors to index lists
    ngram_list = []
    for i in vec.toarray():
        li = []
        for j in range(len(i)):
            if i[j] == 1:
                li.append(j)
        ngram_list.append(li)
        
    return ngram_list
