import pickle
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Any, Optional
from tqdm import tqdm
from gensim.parsing.preprocessing import remove_stopword_tokens
from gensim.models.phrases import Phrases


def lowercasing(x: Any) -> Any:
    """Convert input to lowercase, handling different data types
    
    Args:
        x: Input data which can be a string, list, tuple, or other convertible type
        
    Returns:
        The lowercase version of the input with the same structure
        
    Raises:
        Exception: If the input cannot be converted to lowercase
    """
    if isinstance(x, (list, tuple)):
        x = [lowercasing(_) for _ in x]
    elif isinstance(x, str):
        x = x.lower()
    else:
        try:
            x = str(x).lower()
        except Exception as e:
            raise Exception("Bugs") from e
    return x


def split_sentence(x: str) -> List[str]:
    """Split text into sentences
    
    Args:
        x: Input text string
        
    Returns:
        List of sentences extracted from the input text
    """
    if ". " in x:
        new_x = x.split(". ")
    else:
        new_x = [x]
    last_word = new_x[-1]
    if re.match(r".*\.", last_word) is not None:
        new_x[-1] = last_word.rstrip(".")
    return new_x


def split_word(x: List[str]) -> List[List[str]]:
    """Split sentences into words
    
    Args:
        x: List of sentences
        
    Returns:
        List of lists, where each inner list contains the words from a sentence
    """
    new_x = []
    for _1 in x:
        new_x.append([_2 for _2 in _1.split() if _2 != ""])
    return new_x


def cleanups(x: List[List[str]]) -> List[List[str]]:
    """Remove stopwords from each sentence
    
    Args:
        x: List of sentences, where each sentence is a list of words
        
    Returns:
        List of sentences with stopwords removed
    """
    new_x = []
    for sentence in x:
        new_sentence = remove_stopword_tokens(sentence)
        new_x.append(new_sentence)
    return new_x


def phrasing(x: List[List[str]], phrase_list: List[str], connector: str = "_") -> List[List[str]]:
    """Replace phrases in sentences with connected versions
    
    Args:
        x: List of sentences, where each sentence is a list of words
        phrase_list: List of phrases to look for in the sentences
        connector: String to use for connecting words in a phrase
        
    Returns:
        List of sentences with phrases connected using the connector
    """
    phrase_temp = lowercasing(phrase_list)
    phrase_temp = tuple([tuple(phrase.split()) for phrase in phrase_temp])

    new_x = []
    for sentence in x:
        check_list = []
        append_list = []
        for j in range(len(sentence) + 2):
            for phrase in phrase_temp:
                if j + len(phrase) > len(sentence):
                    continue
                try:
                    bool_list = [bool(re.search('^' + re.escape(phrase_word), word)) or bool(re.search(re.escape(phrase_word) + '$', word)) 
                                 for phrase_word, word in zip(list(phrase), sentence[j:j + len(phrase)])]
                    if np.prod(bool_list) != 0:
                        if j not in append_list and j + len(phrase) not in append_list:
                            check_list.append((j, j + len(phrase), connector.join(sentence[j:j + len(phrase)])))
                            append_list += list(range(j, j + len(phrase)))
                except:
                    print(phrase)
                    print(sentence)

        new_sentence = []
        new_sentence += sentence

        check_list = list(set(check_list))
        check_list = sorted(check_list, key=lambda x: x[0])

        for i, j, phrase in reversed(check_list):
            new_sentence.insert(i, phrase)
            for _ in range(j, i, -1):
                try:
                    del new_sentence[_]
                except Exception as e:
                    print(sentence)
                    print(new_sentence)
                    raise e
        new_x.append(new_sentence)

    return new_x

def phrase(x: List[List[str]], min_count: int, threshold: float) -> List[List[str]]:
    """Generate phrases using gensim's Phrases model
    
    Args:
        x: List of sentences, where each sentence is a list of words
        min_count: Minimum count of word occurrences to be considered for phrasing
        threshold: Score threshold for phrase formation; higher means fewer phrases
        
    Returns:
        List of sentences with automatically detected phrases
    """
    a = Phrases(x, min_count=min_count, threshold=threshold)
    c = Phrases(a[x], min_count=min_count, threshold=threshold)
    d = list(c[a[x]])
    return d
