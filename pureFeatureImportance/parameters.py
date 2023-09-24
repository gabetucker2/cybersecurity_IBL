# SCRIPTS
import datasets
import functions_modelBased
import functions_classBased

# STATIC
DATASET = datasets.dataset_USNW_NB15
ANALYSIS_FUNCTION = functions_classBased.classBased_selectKBest
OUTPUT_NAME = "importanceGraph"

OUTPUT_FOLDER = "graphs"
SHOW_FEATURE_COUNT = 20 # -1 means all features
