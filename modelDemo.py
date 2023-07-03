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

def train(dataset, dataset_name):

    print(f"BEGINNING TRAINING THE {dataset_name} DATASET")

    total_trials = len(dataset[:, 0])

    for trials in range(total_trials):

        training_inputs = train_data[trials, INPUT_COL_IDXS]
        training_output = train_data[trials, OUTPUT_COL_IDX]

        encode_chunk(training_inputs, training_output)
    
    print(f"FINISHED TRAINING THE {dataset_name} DATASET")

def test(dataset, dataset_name):

    print(f"BEGINNING TESTING THE {dataset_name} DATASET")

    trial_errors = 0
    total_trials = len(dataset[:, 0])

    for trials in range(total_trials):

        testing_inputs = test_data[trials, INPUT_COL_IDXS]
        testing_actual = test_data[trials, OUTPUT_COL_IDX]
        testing_predicted = decode_chunk(testing_inputs)

        trial_errors += testing_actual == testing_predicted
    
    error_probability = trial_errors / total_trials
    
    print(f"Accuracy: {(error_probability * 100)}%")
    print(f"FINISHED TESTING THE {dataset_name} DATASET")


# MAIN ROUTINE
train(train_data, "training")
test(test_data, "testing")
