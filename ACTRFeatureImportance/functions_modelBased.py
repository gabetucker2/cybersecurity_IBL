# LIBRARIES
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os
from sklearn.feature_selection import SelectKBest, chi2
import plotly.offline as pyo
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# SCRIPTS
import parameters
import functions_preprocessing

def modelBased_selectKBest():
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
    
def modelBased_randomForest():
    df = functions_preprocessing.load_data()
    X_train, X_test, y_train, y_test = functions_preprocessing.split_data(df)
    X_train, X_test, feature_names = functions_preprocessing.one_hot_encode_and_scale(X_train, X_test, df)

    model_performance = pd.DataFrame(columns=['Accuracy','Recall','Precision','F1-Score','time to train','time to predict','total time'])
    start = time.time()
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0, bootstrap=True).fit(X_train, y_train)
    end_train = time.time()
    y_predictions = model.predict(X_test)
    end_predict = time.time()
    
    accuracy = accuracy_score(y_test, y_predictions)
    recall = recall_score(y_test, y_predictions, average='weighted')
    precision = precision_score(y_test, y_predictions, average='weighted')
    f1s = f1_score(y_test, y_predictions, average='weighted')
    model_performance.loc['Random Forest'] = [accuracy, recall, precision, f1s, end_train-start, end_predict-end_train, end_predict-start]

    # Group the importances by original feature name
    feat_importances = pd.Series(model.feature_importances_, index=feature_names)
    aggregated_importances = {}

    # This function will return the original feature name given the one-hot encoded name
    def get_original_feature_name(encoded_name):
        if 'encoder__' in encoded_name:
            return encoded_name.split('__')[1].split('_')[0]
        elif 'remainder__' in encoded_name:
            return encoded_name.replace('remainder__', '')
        return encoded_name

    for label, importance in feat_importances.items():
        original_name = get_original_feature_name(label)
        
        if original_name in aggregated_importances:
            aggregated_importances[original_name] += importance
        else:
            aggregated_importances[original_name] = importance

    df_scores = pd.DataFrame(list(aggregated_importances.values()), columns=['Importance'])
    df_col = pd.DataFrame(list(aggregated_importances.keys()), columns=['Feature'])

    return df_scores, df_col
