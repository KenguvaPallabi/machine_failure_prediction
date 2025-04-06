import pandas as pd
import joblib

def load_model(path):
    return joblib.load(path)

def load_data(path):
    return pd.read_csv(path)