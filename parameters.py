# LIBRARIES
import pyactup
import pandas

# TRAINING SETS
SMALL_TRAINING_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/Small%20Training%20Set.csv"
BIG_TRAINING_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.csv"
TESTING_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.csv"
OUTPUT_COL_IDX = 41
OUTPUT_NAME = "threat_category"

##############################################################

# PARAMETERS
MEMORY_NOISE = 0.1
MISMATCH_PENALTY = 0.5
SIMILARITY_WEIGHT = 3

##############################################################

# PROCEDURAL DATA
small_train_data = pandas.read_csv(SMALL_TRAINING_URL, header=None).values
big_train_data = pandas.read_csv(BIG_TRAINING_URL, header=None).values
test_data = pandas.read_csv(TESTING_URL, header=None).values
memory = pyactup.Memory(noise=MEMORY_NOISE)
INPUT_COL_IDXS = [i for i in range(len(small_train_data[0, :])) if i != OUTPUT_COL_IDX]
