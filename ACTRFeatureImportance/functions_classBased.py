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

def classBased_selectKBest(X_train, X_test, y_train, y_test, model_performance, feature_names):

    # Load data
    df = parameters.DATASET["combined_inputs"]

    # Numeric preprocessing
    df_numeric = df.select_dtypes(include=[np.number])
    for feature in df_numeric.columns:
        if df_numeric[feature].max() > 10 * df_numeric[feature].median() and df_numeric[feature].max() > 10:
            df[feature] = np.where(df[feature] < df[feature].quantile(0.95), df[feature], df[feature].quantile(0.95))

    df_numeric = df.select_dtypes(include=[np.number])
    for feature in df_numeric.columns:
        if df_numeric[feature].nunique() > 50:
            if df_numeric[feature].min() == 0:
                df[feature] = np.log(df[feature] + 1)
            else:
                df[feature] = np.log(df[feature])

    # Categorical preprocessing
    df_cat = df.select_dtypes(exclude=[np.number])
    for feature in df_cat.columns:
        if df_cat[feature].nunique() > 6:
            df[feature] = np.where(df[feature].isin(df[feature].value_counts().head().index), df[feature], '-')

    # Adjust labels for class-based feature selection if target_class is provided
    X = df.iloc[:, 4:-2]
    y = df.iloc[:, -1]
    target_class_name = parameters.DATASET["binary_target"]
    y = (y == target_class_name).astype(int)  # Convert y labels to binary based on target class

    # Feature Selection
    best_features = SelectKBest(score_func=chi2, k='all')
    fit = best_features.fit(X, y)

    df_scores = pd.DataFrame(fit.scores_)
    df_col = pd.DataFrame(X.columns)

    return df_scores, df_col
