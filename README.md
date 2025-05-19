# FpDoc2vec
Research code for the FpDoc2vec model that predicts chemical roles from database descriptions using NLP. Code used in the paper ['Predicting Chemical Roles from Database Descriptions Using Natural Language Processing'](URL here)

# Summary
- This model has 

# Installation
if you instantly use this model, run the below code.

this repository requires these packages:
<package list>

And this repository is installed by this code
<install code>

# Usage
This model has two steps to prediction.

1. Learn Language Dataset
first, you should prepare the language dataset, like this format;
<todo Table write>

Next, you train FpDoc2Vec model and save it. the below is example code.
<todo write code>

2. Train Activity Dataset
You should prepare the objective activity dataset, which is pairs of the mol onto the activity.
Next, you train FpDoc2Vec-derivate model, which model is exchanging fingerprints to embeddings of FpDoc2Vec and learning embeddings as input and activities as outputs.
Example code is shown below;
<todo write code>

And if you want feature analysis, you can run the SHAP analysis code in result/SHAP directory.
Example code is shown below;
<todo write code>

We supported calcuration of fingerprints importances. So if you want to look graphical interpretations, you should write mapping codes.
Our repository has only one example of mapping, which is atom- and bond-based importance mapping.
<todo example code and figures>

# Performance
Please look up performances from the paper ['Predicting Chemical Roles from Database Descriptions Using Natural Language Processing'](URL here)

# Contact
Please email to gotoh-hiroaki-yw\[at\]ynu.ac.jp if you have any questions or comments.
