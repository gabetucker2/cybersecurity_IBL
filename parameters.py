# LIBRARIES
import pyactup

# SCRIPTS
import functions_decode
import datasets

# STATIC
DATASET = datasets.dataset_NSL_KDD
OUTPUT_NAME = "threat category"
MEMORY_NOISE = 0.1
MISMATCH_PENALTY = 0.5
SIMILARITY_WEIGHT = 1
DECODE_FUNCTION = functions_decode.decode_chunk_blend

EPOCHS = 3

THREATS_PER_TYPE = 3 # -1 means all training data
PROBABILITY_TEST = 0.01

READ_TIME = 0

# PROCEDURAL
memory = pyactup.Memory(noise=MEMORY_NOISE)
