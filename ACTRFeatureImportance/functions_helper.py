# LIBRARIES
import os
import plotly.offline as pyo
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio
import plotly.express as px
import numpy as np

# SCRIPTS
import parameters
import functions_helper

# FUNCTIONS
def get_percent(probability):
    return str(round(probability * 100, 2)) + '%'

def array_to_dictionary(arr):
    return {str(i): value for i, value in enumerate(arr)}

def getOutputDir():
    # Define the full path for the output folder
    output_folder_path = os.path.join(os.getcwd(), parameters.OUTPUT_FOLDER)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Return the desired output name
    return os.path.join(output_folder_path, getOutputName() + ".html")    

def getOutputName():
    # Return the desired output name
    return f"{parameters.ANALYSIS_FUNCTION.__name__}_{parameters.DATASET['name']}"

def plotFeatures(df_scores, df_col):
    
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

    feature_score=pd.concat([df_col,df_scores],axis=1)
    feature_score.columns=['feature','score']
    feature_score.sort_values(by=['score'],ascending=True,inplace=True)

    fig = go.Figure(go.Bar(
                x=feature_score['score'][0:(parameters.SHOW_FEATURE_COUNT)],
                y=feature_score['feature'][0:(parameters.SHOW_FEATURE_COUNT)],
                orientation='h'))

    fig.update_layout(title=f"Top {parameters.SHOW_FEATURE_COUNT} Features in `{functions_helper.getOutputName()}`",
                    height=1200,
                    showlegend=False,
                    )
    
    # Concatenate to form the absolute path to save the plot
    pyo.plot(fig, filename=getOutputDir())
