import re
import numpy as np
import pandas as pd
from typing import List, Any
from tqdm import tqdm
from gensim.parsing.preprocessing import remove_stopword_tokens
from gensim.models.phrases import Phrases


def lowercasing(x: Any) -> Any:
    """Convert input to lowercase."""
    if isinstance(x, (list, tuple)):
        x = [lowercasing(_) for _ in x]
    elif isinstance(x, str):
        x = x.lower()
    else:
        try:
            x = str(x).lower()
        except Exception as e:
            raise Exception(f"Failed to lowercase value of type {type(x)}: {e}") from e
    return x


def split_sentence(x: str) -> List[str]:
    """Split text into a list of sentences."""
    if ". " in x:
        new_x = x.split(". ")
    else:
        new_x = [x]
    last_word = new_x[-1]
    if re.match(r".*\.", last_word) is not None:
        new_x[-1] = last_word.rstrip(".")
    return new_x


def split_word(x: List[str]) -> List[List[str]]:
    """Split a list of sentences into lists of words."""
    new_x = []
    for _1 in x:
        new_x.append([_2 for _2 in _1.split() if _2 != ""])
    return new_x


def cleanups(x: List[List[str]]) -> List[List[str]]:
    """Remove stopwords from each sentence."""
    new_x = []
    for sentence in x:
        new_sentence = remove_stopword_tokens(sentence)
        new_x.append(new_sentence)
    return new_x


def phrasing(x: List[List[str]], phrase_list: List[str], connector: str = "_") -> List[List[str]]:
    """Connect multi-word phrases in sentences using a connector string."""
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
    """Detect and merge bigrams/trigrams using gensim's Phrases model."""
    bigrams = Phrases(x, min_count=min_count, threshold=threshold)
    trigrams = Phrases(bigrams[x], min_count=min_count, threshold=threshold)
    return list(trigrams[bigrams[x]])


def main_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text preprocessing to compound descriptions."""
    df["description_lower"] = df["description"].map(lambda x: lowercasing(x))
    df["description_split_sentence"] = df["description_lower"].map(lambda x: split_sentence(x))
    df["description_split"] = df["description_split_sentence"].map(lambda x: split_word(x))
    df["description_remove_stop_words"] = df["description_split"].map(lambda x: cleanups(x))

    li = []
    for i in tqdm(range(len(df))):
        li.append(phrasing(df.at[i, "description_remove_stop_words"], phrase_list=[df.at[i, "NAME"]]))
    df["description_phrases"] = li
    df["description_phrases"] = df["description_phrases"].map(lambda x: phrase(x, 1, 0.7))

    df["description_gensim"] = df["description_remove_stop_words"].map(lambda x: phrase(x, 1, 0.7))

    return df
