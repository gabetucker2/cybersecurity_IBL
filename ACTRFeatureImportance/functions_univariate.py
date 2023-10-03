# LIBRARIES
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

# SCRIPTS
import functions_preprocessing

def univariate_selectKBest():
    df = functions_preprocessing.load_data()
    df = functions_preprocessing.numeric_preprocessing(df)
    df = functions_preprocessing.categorical_preprocessing(df)

    best_features = SelectKBest(score_func=chi2, k='all')
    X = df.iloc[:,4:-2]
    y = df.iloc[:,-1]
    fit = best_features.fit(X, y)

    df_scores = pd.DataFrame(fit.scores_)
    df_col = pd.DataFrame(X.columns)

    return df_scores, df_col
