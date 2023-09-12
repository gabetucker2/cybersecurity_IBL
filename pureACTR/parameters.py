# LIBRARIES
import pyactup

# SCRIPTS
import datasets
import functions_decode

# STATIC
DATASET = datasets.dataset_USNW_NB15
BINARY = False
OUTPUT_NAME = "threat category"
MEMORY_NOISE = 0.1
MISMATCH_PENALTY = 0.5
SIMILARITY_WEIGHT = 1
DECODE_FUNCTION = functions_decode.decode_chunk_retrieval_partial

EPOCHS = 10

THREATS_PER_TYPE = 1 # -1 means all training data
PROBABILITY_TEST = 0.00025

READ_TRAIN_TIME = 0
READ_TEST_TIME = 0

# PROCEDURAL
memory = pyactup.Memory(noise=MEMORY_NOISE)
