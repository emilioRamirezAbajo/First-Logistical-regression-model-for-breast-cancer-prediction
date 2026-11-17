import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ucimlrepo import fetch_ucirepo, list_available_datasets 
from sklearn import linear_model

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# fetch dataset 
breast_cancer = fetch_ucirepo(id=17)

# Separate features and target variable
dX = breast_cancer.data.features
dy = breast_cancer.data.targets.values.ravel()  # Flatten to 1D array


regression_model = linear_model.LogisticRegression(max_iter = 1000)
regression_model.fit(dX, dy)

# Make a prediction 
predicted_cancer = regression_model.predict(dX)

print(predicted_cancer)
