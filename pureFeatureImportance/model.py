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
import time

# SCRIPTS
import parameters
import functions_helper

time.sleep(0.1) # allow other scripts to run first

warnings.filterwarnings("ignore") # ignore redundant warnings

print(f"RUNNING `{parameters.ANALYSIS_FUNCTION.__name__}` FUNCTION")

X_train, X_test, y_train, y_test, model_performance, feature_names = functions_helper.getAnalysisInputs()
df_scores, df_col = parameters.ANALYSIS_FUNCTION(X_train, X_test, y_train, y_test, model_performance, feature_names) # call main function

print(f"FINISHED `{parameters.ANALYSIS_FUNCTION.__name__}` FUNCTION")

functions_helper.plotFeatures(df_scores, df_col)

print(f"CREATED GRAPH AT `{functions_helper.getOutputName()}`")
