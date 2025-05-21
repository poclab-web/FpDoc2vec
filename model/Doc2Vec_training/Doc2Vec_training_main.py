from Doc2Vec_training_function import generate_morgan_fingerprints, lowercasing, exact_name, train_doc2vec_model, main_doc2vec


# Example usage - replace with your actual params
# Please change the parameter values as you like.
param = {"vector_size": 100, 
     "min_count": 0,
     "window": 10,
     "min_alpha": 0.023491749982816976,
     "sample": 7.343338709169564e-06,
     "epochs": 859,
     "negative": 2,
     "ns_exponent": 0.8998927133390002,
     "workers": 1, 
     "seed": 100}

# Building FpDoc2Vec model
# Example usage - replace with your actual file paths
input_file = "10genre_dataset.pkl"
output_model_name = "fpdoc2vec.model"
finger_list = list(df["fp_3_4096"])
main_doc2vec(input_file, output_model_name, param, finger_list)

# Building NameDoc2Vec model
# Extract compound names
allcompounds = exact_name(df)
# Example usage - replace with your actual file paths
output_model_name = "namedoc2vec.model"
main_doc2vec(input_file, output_model_name, param, allcompounds)

# Building FpDoc2Vec model for unseen compounds
# Generate fingerprints
input_file = "train_df.pkl"
with open(input_file, "rb") as f:
     df = pickle.load(f)
finger_list = list(df["fp_3_4096"])

# Example usage - replace with your actual file paths
output_model_name = "fpdoc2vec_unseen.model"
main_doc2vec(input_file, output_model_name, param, finger_list)
