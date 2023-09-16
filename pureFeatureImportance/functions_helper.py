# LIBRARIES
import os
import plotly.offline as pyo
import plotly.graph_objects as go
import pandas as pd

# SCRIPTS
import parameters

def getOutputName():
    return f"{parameters.OUTPUT_FOLDER}/{parameters.OUTPUT_NAME}_{parameters.DATASET['name']}.html"

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
    
    # Get current working directory
    current_directory = os.getcwd()

    # Concatenate to form the absolute path to save the plot
    pyo.plot(fig, filename=getOutputName())
    