# LIBRARIES
import os
import plotly.offline as pyo
import plotly.graph_objects as go
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np

# SCRIPTS
import parameters

def getAnalysisInputs():
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

    return X_train, X_test, y_train, y_test, model_performance, feature_names

def getOutputName():
    return f"/{parameters.OUTPUT_FOLDER}/{parameters.OUTPUT_NAME}_{parameters.DATASET['name']}.html"

def plotFeatures(df_scores, df_col):

    feature_score=pd.concat([df_col,df_scores],axis=1)
    feature_score.columns=['feature','score']
    feature_score.sort_values(by=['score'],ascending=True,inplace=True)

    fig = go.Figure(go.Bar(
                x=feature_score['score'][0:(parameters.SHOW_FEATURE_COUNT)],
                y=feature_score['feature'][0:(parameters.SHOW_FEATURE_COUNT)],
                orientation='h'))

    fig.update_layout(title=f"Top {parameters.SHOW_FEATURE_COUNT} Features",
                    height=1200,
                    showlegend=False,
                    )
    
    # Concatenate to form the absolute path to save the plot
    pyo.plot(fig, filename=os.getcwd()+getOutputName())
