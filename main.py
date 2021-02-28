# architecture
## load reasonable dataset -> encrypt the dataset such that it is homomorphically parsable
## build homomorphic encryption enabled model -> obtain accuracy on performing inference on HE-based data
## build benchmark model with original data -> obtain accuracy based on non-HE data

# dataset: credit scoring benchmark dataset (credit default, client characteristics dataset)
# https://archive.ics.uci.edu/ml/machine-learning-databases/00350/

import pandas as pd
from utils import dataload
from model import vanillaModel, homomorphicEncryptionModel, BenmarkModel

cci_data = pd.read_csv("uci_cci.csv")

input_cols = list(cci_data.columns)[:-1]
output_cols = list(cci_data.columns)[-1]

x_data, y_data, x_data_shortened, y_data_shortened, X_enc, y_enc, H_enc = dataload(cci_data, input_cols, output_cols)

# https://www.cs.cmu.edu/~rjhall/JOS_revised_May_31a.pdf
# source for use of orthogonal matrices and invertible matrices transformation for homomorophic encryption

# Test case 1: Vanilla model (benchmark credit rating/scoring model for default prediction)
# vanillaModel(x_data, y_data)

# BenmarkModel(x_data, y_data, x_data_shortened)
# Test case 2: Homomorphic encryption model running on encrypted data
homomorphicEncryptionModel(X_enc, y_enc, x_data, H_enc)


# Use Unencrpted data
BenmarkModel(cci_data)