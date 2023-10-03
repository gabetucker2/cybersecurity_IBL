# LIBRARIES
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif

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

    # Check if data has negative values to decide on the scoring function
    if (X < 0).any().any():
        score_func = f_classif
    else:
        score_func = chi2

    best_features = SelectKBest(score_func=score_func, k='all')
    fit = best_features.fit(X, y)

    df_scores = pd.DataFrame(fit.scores_)
    df_col = pd.DataFrame(X.columns)

    return df_scores, df_col
