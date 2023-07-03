# LIBRARIES
import pyactup
import pandas

# PARAMETERS
TRAINING_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.csv"
TESTING_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.csv"
OUTPUT_COL_IDX = 41
OUTPUT_NAME = "threat category"

NOISE = 0.1

# VARIABLES
train_data = pandas.read_csv(TRAINING_URL).values
test_data = pandas.read_csv(TESTING_URL).values
memory = pyactup.Memory(noise=NOISE)
num_cols = len(train_data[0, :])
testing_errors = 0

INPUT_COL_IDXS = [i for i in range(num_cols) if i != OUTPUT_COL_IDX]

# FUNCTIONS
def arrays_to_dict(*arrs):
    # examples:
    # arrays_to_dict(["apple", "banana", "grape"]) => {0 : "apple", 1 : "banana", 2: "grape"}
    # arrays_to_dict(["apple", "banana"], ["grape", "orange"]) => {0 : "apple", 1 : "banana", 2: "grape", 3 : "orange"}

    args = {}
    i = 0
    for array in arrs:
        for value in array:
            args[i] = value
            i += 1
    return args


def encode_chunk(inputs, output):
    memory.learn(arrays_to_dict(inputs, output))

def decode_chunk(inputs):
    return (memory.retrieve(arrays_to_dict(inputs))).get(OUTPUT_NAME)

def train(dataset):
    for row_idx in range(len(dataset[:, 0])):

        training_inputs = train_data[row_idx, INPUT_COL_IDXS]
        training_output = train_data[row_idx, OUTPUT_COL_IDX]

        encode_chunk(training_inputs, training_output)

def test(dataset):
    for row_idx in range(len(dataset[:, 0])):

        testing_inputs = test_data[row_idx, INPUT_COL_IDXS]
        testing_actual = test_data[row_idx, OUTPUT_COL_IDX]
        testing_predicted = decode_chunk(testing_inputs)

# MAIN ROUTINE
# train(train_data)
# test(test_data)
