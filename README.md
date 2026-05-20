[![DOI](https://zenodo.org/badge/981953595.svg)](https://doi.org/10.5281/zenodo.19656925)

# FpDoc2vec
Research code for the FpDoc2vec model that predicts chemical roles from database descriptions using NLP. Code used in the paper ['Predicting Chemical Roles from Database Descriptions Using Natural Language Processing'](URL here)

# Summary
- This model connects two kinds of information, which are linguistic and chemical.
- Explanatory variables created by this model is more useful than RDKit descriptors and fingerprints.

# Installation
this repository requires these packages:
numpy==1.24.4
pandas==1.5.3
scikit-learn==1.2.2
matplotlib==3.6.0
seaborn==0.13.0
gensim==4.3.2
lightgbm==3.3.5
xgboost==2.1.3
shap==0.44.1
umap-learn==0.5.5
optuna==4.1.0
optunahub==0.2.0
rdkit==2022.3.5
pubchempy==1.0.4
requests==2.31.0
beautifulsoup4==4.12.2
lxml==4.9.3

And this repository is installed by this prompt code by Anaconda
```
conda install requirements.txt
```

# Usage
This model has two steps to prediction.

1. Learn Language Dataset
first, you should prepare the language dataset, like this format;

| ROMol                          | Description                                       | Objective                                                                               | 
| ------------------------------ | ------------------------------------------------- | --------------------------------------------------------------------------------------- | 
| This cells contain MOL object  | This cells describe the compound by string object | This cells contain Your objective variables (such as toxicity or antioxidant activitiy) | 

Next, you train FpDoc2Vec model and save it. the below is example python code.
```
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from utils import load_pickle, generate_ecfp_fingerprints, build_tagged_corpus
from config import doc2vec_params as DOC2VEC_PARAMS

dataset = load_pickle("your dataset.pkl")

# Generate fingerprints
_, bit_list = generate_ecfp_fingerprints(list(dataset["ROMol"]), radius=3, n_bits=4096)  # Change radius and bits as you want
corpus = build_tagged_corpus(dataset, bit_list, "description_column_name")

model = Doc2Vec(corpus, **DOC2VEC_PARAMS)
model.save("your model name.model")
```

2. Train Activity Dataset
Next, you train FpDoc2Vec-derivate model, which model is exchanging fingerprints to embeddings of FpDoc2Vec and learning embeddings as input and activities as outputs.
Example code is shown below;
```
import lightgbm as lgb
from gensim.models.doc2vec import Doc2Vec
from utils import load_pickle, generate_ecfp_fingerprints, fingerprints_to_vectors, main_cv

dataset = load_pickle("your dataset.pkl")
fp_model = Doc2Vec.load("your model name.model")

with open("your predict conditions.pkl", "rb") as f:
  conditions = pickle.load(f)
classifier = lgb.LGBMClassifier(**conditions)

_, bit_list = generate_ecfp_fingerprints(list(dataset["ROMol"]), radius=3, n_bits=4096)
fpvec = fingerprints_to_vectors(bit_list, fp_model)

results = main_cv(df=dataset, X_vec=fpvec, classifier=classifier)
```

And if you want feature analysis, you can run the SHAP analysis code in result/SHAP directory.
Example code is shown below;
```
import pickle
import numpy as np
import shap
import lightgbm as lgb
from gensim.models import Doc2Vec
from utils import load_pickle, save_pickle, generate_ecfp_fingerprints
from result.SHAP.calculate_core import shap_variables

# Load Data and models
dataset = load_pickle("your activity dataset.pkl")
doc2vec_model = Doc2Vec.load("your model name.model")

with open("your conditions.pkl", "rb") as f:
  conditions = pickle.load(f)

# Define variables (Change as you want)
purpose = "antioxidant"
max_evals = 200000

# SHAP preparation
fingerprints, _ = generate_ecfp_fingerprints(list(dataset["ROMol"]), radius=3, n_bits=4096)
y = (dataset[purpose] == purpose).astype(int).to_numpy()
lightgbm_model = lgb.LGBMClassifier(**conditions)
pipeline, masker = shap_variables(doc2vec_model.dv.vectors, lightgbm_model, mask='xor')
pipeline.fit(fingerprints, y)

# SHAP calculation
explainer = shap.Explainer(lambda x: pipeline.predict_proba(x)[:, 1], masker=masker)
value = explainer(fingerprints, max_evals)

save_pickle(value, "your shap file.pkl")
```

We supported calcuration of fingerprints importances. So if you want to look graphical interpretations, you should write mapping codes.
Our repository has only one example of mapping, which is atom- and bond-based importance mapping.
```
from utils import load_pickle
from result.SHAP.value_to_structure_core import visualize_shap_on_molecule

shap_values = load_pickle("your shap file.pkl")
dataset = load_pickle("your activity dataset.pkl")

# View mapping (Change arguments as you want)
result_svg = visualize_shap_on_molecule(
    compound_name="your compound name",
    df=dataset,
    shap_values=shap_values,
    radius=3,
    nBits=4096,
    compound_column="NAME",
    mol_column="ROMol",
    output="output.svg"
)
```

# Other Details
If you want other details like performances, please look the paper ['Predicting Chemical Roles from Database Descriptions Using Natural Language Processing'](URL here)

# Data Source
This repository uses [ChEBI Database](https://www.ebi.ac.uk/chebi/) and [PubChem Database](https://pubchem.ncbi.nlm.nih.gov/).

# Contact
Please email to gotoh-hiroaki-yw\[at\]ynu.ac.jp if you have any questions or comments.
