# LIBRARIES
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
import os
from sklearn.feature_selection import SelectKBest, chi2
import plotly.offline as pyo
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# SCRIPTS
import parameters

def selectKBest(X_train, X_test, y_train, y_test, model_performance, feature_names):
    # Configure display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.expand_frame_repr', False)

    # Configure plotly templates
    pio.templates["ck_template"] = go.layout.Template(
        layout_colorway=px.colors.sequential.Viridis,
        layout_autosize=False,
        layout_width=800,
        layout_height=600,
        layout_font=dict(family="Calibri Light"),
        layout_title_font=dict(family="Calibri"),
        layout_hoverlabel_font=dict(family="Calibri Light"),
    )
    pio.templates.default = 'ck_template+gridon'

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
    
def randomForest(X_train, X_test, y_train, y_test, model_performance, feature_names):
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
