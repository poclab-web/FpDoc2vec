# FpDoc2vec
Research code for the FpDoc2vec model that predicts chemical roles from database descriptions using NLP. Code used in the paper ['Predicting Chemical Roles from Database Descriptions Using Natural Language Processing'](URL here)

# Summary
- This model has 

# Installation
this repository requires these packages:
<package list>

And this repository is installed by this prompt code
```
pip install requirements.txt
```

# Usage
This model has two steps to prediction.

1. Learn Language Dataset
first, you should prepare the language dataset, like this format;
<todo Table write>

Next, you train FpDoc2Vec model and save it. the below is example python code.
```
import ~~

with open("your dataset.pkl", "rb") as f:
  contents = pickle.load(f)
with open("your conditions.pkl", "rb") as f:
  conditions = pickle.load(f)

model = "make model function"(contents, conditions=conditions)

model.save("your model name.d2v")
```

2. Train Activity Dataset
You should prepare the objective activity dataset, which is pairs of the mol onto the activity.
Next, you train FpDoc2Vec-derivate model, which model is exchanging fingerprints to embeddings of FpDoc2Vec and learning embeddings as input and activities as outputs.
Example code is shown below;
```
import ~~

fpdoc2vec = model.load("your model name.d2v")
with open("your activities.pkl", "rb") as f:
  dataset = pickle.load(f)
with open("your conditions.pkl", "rb") as f:
  conditions = pickle.load(f)

model = "predict activity function"(dataset, fpdoc2vec, conditions=conditions)
model.save("your model name.prd")
"plot function"(model)
```

And if you want feature analysis, you can run the SHAP analysis code in result/SHAP directory.
Example code is shown below;
```
import ~~

prd_model = model.load("your prediction model name.prd")
fpdoc2vec = model.load("your model name.d2v")
with open("your activity dataset.pkl", "rb") as f:
  dataset = pickle.load(f)
with open("your conditions.pkl", "rb") as f:
  conditions = pickle.load(f)

shaps = "make SHAP function"(dataset, prd_model, fpdoc2model, conditions=conditions)
"plot function"(shap)

with open("your shap file.pkl", "wb") as f:
  pickle.dump(shap, f)
```

We supported calcuration of fingerprints importances. So if you want to look graphical interpretations, you should write mapping codes.
Our repository has only one example of mapping, which is atom- and bond-based importance mapping.
```
import ~~

with open("your shap file.pkl", "rb") as f:
  shap = pickle.load(f)
with open("your activity dataset, "rb") as f:
  dataset = pickle.load(f)

"plot function"(shap, dataset)
```

# Performance
Please look up performances from the paper ['Predicting Chemical Roles from Database Descriptions Using Natural Language Processing'](URL here)

# Contact
Please email to gotoh-hiroaki-yw\[at\]ynu.ac.jp if you have any questions or comments.
