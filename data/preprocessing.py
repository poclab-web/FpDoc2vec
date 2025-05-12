import pickle
import re
import pandas as pd
from gensim.parsing.preprocessing import remove_stopword_tokens

def lowercasing(x):
    """Convert input to lowercase, handling different data types"""
    if type(x) == list or type(x) == tuple:
        x = [lowercasing(_) for _ in x]
    elif type(x) == str:
        x = x.lower()
    else:
        try:
            x = str(x).lower()
        except Exception as e:
            raise e("Bugs")
    return x

def split_sentence(x):
    """Split text into sentences"""
    if ". " in x:
        new_x = x.split(". ")
    else:
        new_x = [x]
    last_word = new_x[-1]
    if re.match(r".*\.", last_word) != None:
        new_x[-1] = last_word.rstrip(".")
    return new_x

def split_word(x):
    """Split sentences into words"""
    new_x = []
    for _1 in x:
        new_x.append([_2 for _2 in _1.split() if _2 != ""])
    return new_x

def cleanups(x):
    """Remove stopwords from each sentence"""
    new_x = []
    for sentence in x:
        new_sentence = remove_stopword_tokens(sentence)
        new_x.append(new_sentence)
    return new_x

def preprocess_chemical_descriptions():
    """Load and preprocess chemical compound descriptions"""
    # Load the data
    with open("3starAll_ver2.pkl", "rb") as f:
        dict_data = pickle.load(f)
        all_text_df = pd.DataFrame(dict_data.items(), columns=["compounds", "description"])
    
    # Apply preprocessing steps
    all_text_df["description_lower"] = all_text_df["description"].map(lambda x: lowercasing(x))
    all_text_df["description_split_sentence"] = all_text_df["description_lower"].map(lambda x: split_sentence(x))
    all_text_df["description_split"] = all_text_df["description_split_sentence"].map(lambda x: split_word(x))
    all_text_df["description_remove_stop_words"] = all_text_df["description_split"].map(lambda x: cleanups(x))
    
    # Save the processed data
    with open("chemdata/3starAll_text_ver2.pkl", "wb") as f:
        pickle.dump(all_text_df, f)
    
    return all_text_df

if __name__ == "__main__":
    preprocess_chemical_descriptions()
