# LIBRARIES
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import warnings
import plotly.express as px
import os
from sklearn.feature_selection import SelectKBest, chi2
import plotly.offline as pyo

# SCRIPTS
import parameters

# Configure display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.expand_frame_repr', False)

warnings.filterwarnings('ignore')

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

# Displaying the figure using plotly offline

# Get current working directory
current_directory = os.getcwd()

# Concatenate to form the absolute path to save the plot
pyo.plot(fig, filename=f"graphs/featureImportance_{parameters.DATASET['name']}.html")
