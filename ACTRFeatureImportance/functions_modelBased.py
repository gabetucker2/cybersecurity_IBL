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

def modelBased_selectKBest():

    # Load data
    df = parameters.DATASET["combined_inputs"]

    df_numeric = df.select_dtypes(include=[np.number])

    for feature in df_numeric.columns:
        if df_numeric[feature].max()>10*df_numeric[feature].median() and df_numeric[feature].max()>10 :
            df[feature] = np.where(df[feature]<df[feature].quantile(0.95), df[feature], df[feature].quantile(0.95))

    df_numeric = df.select_dtypes(include=[np.number])
    df_before = df_numeric.copy()

    for feature in df_numeric.columns:
        if df_numeric[feature].nunique()>50:
            if df_numeric[feature].min()==0:
                df[feature] = np.log(df[feature]+1)
            else:
                df[feature] = np.log(df[feature])

    df_numeric = df.select_dtypes(include=[np.number])

    df_cat = df.select_dtypes(exclude=[np.number])

    for feature in df_cat.columns:
        if df_cat[feature].nunique()>6:
            df[feature] = np.where(df[feature].isin(df[feature].value_counts().head().index), df[feature], '-')

        df_cat = df.select_dtypes(exclude=[np.number])

    # Feature Selection

    best_features = SelectKBest(score_func=chi2,k='all')

    X = df.iloc[:,4:-2]
    y = df.iloc[:,-1]
    fit = best_features.fit(X,y)

    df_scores=pd.DataFrame(fit.scores_)
    df_col=pd.DataFrame(X.columns)

    return df_scores, df_col
    
def modelBased_randomForest():

    df = parameters.DATASET["combined_inputs"]
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X.head()
    df_cat = df.select_dtypes(exclude=[np.number])
    feature_names = list(X.columns)
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,2,3])], remainder='passthrough')
    X = ct.fit_transform(X).toarray()
    for label in list(df_cat['state'].value_counts().index)[::-1][1:]:
        feature_names.insert(0,label)
        
    for label in list(df_cat['service'].value_counts().index)[::-1][1:]:
        feature_names.insert(0,label)
        
    for label in list(df_cat['proto'].value_counts().index)[::-1][1:]:
        feature_names.insert(0,label)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.2, 
                                                        random_state = 0,
                                                        stratify=y)
    sc = StandardScaler()
    X_train[:, 18:] = sc.fit_transform(X_train[:, 18:])
    X_test[:, 18:] = sc.transform(X_test[:, 18:])
    model_performance = pd.DataFrame(columns=['Accuracy','Recall','Precision','F1-Score','time to train','time to predict','total time'])

    start = time.time()
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0, bootstrap=True).fit(X_train, y_train)
    end_train = time.time()
    y_predictions = model.predict(X_test) # These are the predictions from the test data.
    end_predict = time.time()
    
    accuracy = accuracy_score(y_test, y_predictions)
    recall = recall_score(y_test, y_predictions, average='weighted')
    precision = precision_score(y_test, y_predictions, average='weighted')
    f1s = f1_score(y_test, y_predictions, average='weighted')

    model_performance.loc['Random Forest'] = [accuracy, recall, precision, f1s, end_train-start, end_predict-end_train, end_predict-start]

    # Instead of plotting, we'll prepare the scores and columns for output.
    feat_importances = pd.Series(model.feature_importances_, index=feature_names)
    
    df_scores = pd.DataFrame(feat_importances.values)
    df_col = pd.DataFrame(feat_importances.index)

    return df_scores, df_col
