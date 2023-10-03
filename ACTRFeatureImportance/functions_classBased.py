# LIBRARIES
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os
from sklearn.feature_selection import SelectKBest, chi2
import plotly.offline as pyo
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# SCRIPTS
import parameters
import functions_preprocessing

def classBased_selectKBest():
    df = functions_preprocessing.load_data()
    df = functions_preprocessing.numeric_preprocessing(df)
    df = functions_preprocessing.categorical_preprocessing(df)

    X = df.iloc[:, 4:-2]
    y = df.iloc[:, -1]
    target_class_name = parameters.DATASET["binary_target"]
    y = (y == target_class_name).astype(int)

    best_features = SelectKBest(score_func=chi2, k='all')
    fit = best_features.fit(X, y)

    df_scores = pd.DataFrame(fit.scores_)
    df_col = pd.DataFrame(X.columns)

    return df_scores, df_col
