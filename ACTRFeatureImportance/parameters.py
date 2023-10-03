# LIBRARIES
import pyactup

# SCRIPTS
import datasets
import functions_decode
import functions_univariate
import functions_modelBased
import functions_classBased

# DATASETS
DATASET = datasets.dataset_USNW_NB15
BINARY = False
OUTPUT_NAME = "threat category"
MEMORY_NOISE = 0.1
MISMATCH_PENALTY = 0.5
SIMILARITY_WEIGHT = 1
DECODE_FUNCTION = functions_decode.decode_chunk_blend

# ACT-R
RUN_ACT_UP = False

EPOCHS = 1
THREATS_PER_TYPE = 1                                            # -1 means all training data
PROBABILITY_TEST = 0.01

READ_TRAIN_TIME = 0
READ_TEST_TIME = 0

# FEATURE IMPORTANCE
RUN_FEATURE_IMPORTANCE = True

# FROM functions_univariate, functions_modelBased, or functions_classBased
ANALYSIS_FUNCTION = functions_modelBased.modelBased_randomForest

OUTPUT_FOLDER = "graphs"
SHOW_FEATURE_COUNT = 20 # -1 means all features

# PROCEDURAL
memory = pyactup.Memory(noise=MEMORY_NOISE)
