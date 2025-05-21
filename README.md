[![DOI](svg name)](doi name)

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

And this repository is installed by this prompt code
```
pip install requirements.txt
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
import pickle
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from FpDoc2Vec.model.Do2Vec_training.Doc2Vec_training_function import generate_morgan_fingerprints, lowercasing, exact_name, train_doc2vec_model

with open("your dataset.pkl", "rb") as f:
  dataset = pickle.load(f)
with open("your paramaters.pkl", "rb") as f:
  params = pickle.load(f)

# Generate fingerprints
dataset["FP"] = generate_morgan_fingerprints(df, 3, 4096) # Change radius and bits as you want
finger_list = list(dataset["FP"])

model = train_doc2vec_model(dataset, finger_list, params, "description_column_name")
model.save("your model name.d2v")
```

2. Train Activity Dataset
Next, you train FpDoc2Vec-derivate model, which model is exchanging fingerprints to embeddings of FpDoc2Vec and learning embeddings as input and activities as outputs.
Example code is shown below;
```
import pickle
import pandas as pd
import lightgbm as lgb
from gensim.models.doc2vec import Doc2Vec
from FpDoc2vec.result.full_prediction.FpDoc2Vec import add_vectors, evaluate_category, make_fp2vector

fpdoc2vec = model.load("your model name.d2v")
with open("your dataset.pkl", "rb") as f:
  dataset = pickle.load(f)
with open("your predict conditions.pkl", "rb") as f:
  conditions = pickle.load(f)

fpvec = make_fp2vector(model_path=fp_model_path, df=df)
objectives = ['antioxidant', 'anti_inflammatory_agent']  # Change objectives as you want
lightgbm_model = lgb.LGBMClassifier(**conditions)

results = {}
    for category in categories:
        y = np.array([1 if i == category else 0 for i in dataset[category]])
        results[category] = evaluate_category(category, X_vec, y, lightgbm_model)
model.save("your perdiction model name.prd")
"plot function"(model)
```

And if you want feature analysis, you can run the SHAP analysis code in result/SHAP directory.
Example code is shown below;
```
import pickle
import pandas as pd
import lightgbm as lgb
from gensim.models.doc2vec import Doc2Vec
from FpDoc2vec.result.SHAP.shap_fpdoc2vec import *

# Load Data and models
prd_model = model.load("your prediction model name.prd")
fpdoc2vec = model.load("your model name.d2v")
with open("your activity dataset.pkl", "rb") as f:
  dataset = pickle.load(f)
with open("your conditions.pkl", "rb") as f:
  conditions = pickle.load(f)

# Define variables (Change as you want)
purpose = "antioxidant"
max_evals = 200000

# SHAP preparation
pipeline, masker = shap_variables(fpdoc2vec.dv.vectors, prd_model)
y = np.array([1 if i == purpose else 0 for i in dataset[purpose]])
fingerprint = generate_morgan_fingerprints(dataset, 3, 4096) # Generate by your conditions

# SHAP calcuration
explainer = shap.Explainer(lambda x: pipeline.predict_proba(x)[:, 1], masker=masker)
value = explainer(fingerprint, max_evals)

# Save file
with open("your shap file.pkl", "wb") as f:
  pickle.dump(value, f)
```

We supported calcuration of fingerprints importances. So if you want to look graphical interpretations, you should write mapping codes.
Our repository has only one example of mapping, which is atom- and bond-based importance mapping.
```
import pickle
import pandas as pd
import lightgbm as lgb
from gensim.models.doc2vec import Doc2Vec
from FpDoc2vec.result.SHAP.shapvalue_to_sturucture import *

with open("your shap file.pkl", "rb") as f:
  shap = pickle.load(f)
with open("your activity dataset, "rb") as f:
  dataset = pickle.load(f)

# variables define
mol = dataset["ROMol"]
index = dataset.index[0]
scale_factor = 1.0 # Change as you want
fp_radius = 3 # Set same condition as previous
nBits = 4096 # Set same condition as previous

# View mapping
result_svg = visualize_shap_on_molecule(
        mol=mol, 
        shap_values=shap_values, 
        index=index,
        scale_factor=scale_factor,
        fp_radius=fp_radius,
        nBits=nBits
    )
```

# Other Details
If you want other details like performances, please look the paper ['Predicting Chemical Roles from Database Descriptions Using Natural Language Processing'](URL here)

# Data Source
This repository uses [ChEBI Database](https://www.ebi.ac.uk/chebi/) and [PubChem Database](https://pubchem.ncbi.nlm.nih.gov/).

# Contact
Please email to gotoh-hiroaki-yw\[at\]ynu.ac.jp if you have any questions or comments.
