# Read & laod .pkl & .json files
# Logic to load model and features

import pickle
import json
import numpy as np
import pandas as pd 

# To supress warning
import warnings
warnings.filterwarnings(action='ignore')

def load_model_and_features():
    # Load the trained model
    with open(r'artifacts\lr_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load the feature information
    with open(r'artifacts\feature.json', 'r') as f:
        features = json.load(f)
    
    return model, features


