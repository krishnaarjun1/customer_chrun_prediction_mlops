import joblib
import pandas as pd

def load_model(path):
    return joblib.load(path)

def predict(model, input_df):
    return model.predict(input_df)
