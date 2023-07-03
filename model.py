# LIBRARIES
import pyactup
import pandas
import random

# PARAMETERS
TRAINING_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.csv"
TESTING_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.csv"
OUTPUT_COL_IDX = 41
OUTPUT_NAME = "threat category"

MEMORY_NOISE = 0.1
PROBABILITY_ENCODE_TRIAL = 0.1

# VARIABLES
train_data = pandas.read_csv(TRAINING_URL, header = None).values
test_data = pandas.read_csv(TESTING_URL, header = None).values
memory = pyactup.Memory(noise=MEMORY_NOISE)

INPUT_COL_IDXS = [i for i in range(len(train_data[0, :])) if i != OUTPUT_COL_IDX]

# FUNCTIONS
def encode_chunk(inputs, output):

    data_to_encode = {}

    for i in range(len(inputs)):
        data_to_encode[f'{i}'] = inputs[i]
    
    data_to_encode[OUTPUT_NAME] = output

    memory.learn(data_to_encode)

def decode_chunk(inputs):

    data_to_decode = {}

    for i in range(len(inputs)):
        data_to_decode[f'{i}'] = inputs[i]

    return (memory.retrieve(data_to_decode) or {}).get(OUTPUT_NAME)

def train(dataset, dataset_name):

    print(f"BEGINNING TRAINING THE `{dataset_name}` DATASET")

    total_trials = len(dataset[:, 0])

    for trials in range(total_trials):

        if PROBABILITY_ENCODE_TRIAL < random.random():

            training_inputs = dataset[trials, INPUT_COL_IDXS]
            training_output = dataset[trials, OUTPUT_COL_IDX]

            encode_chunk(training_inputs, training_output)
    
    print(f"FINISHED TRAINING THE `{dataset_name}` DATASET")

def test(dataset, dataset_name):

    print(f"BEGINNING TESTING THE `{dataset_name}` DATASET")

    trial_errors = 0
    total_trials = len(dataset[:, 0])

    for trials in range(total_trials):

        testing_inputs = dataset[trials, INPUT_COL_IDXS]
        testing_actual = dataset[trials, OUTPUT_COL_IDX]
        testing_predicted = decode_chunk(testing_inputs)

        trial_errors += testing_actual == testing_predicted
    
    error_probability = trial_errors / total_trials
    
    print(f"Accuracy: {(error_probability * 100)}%")
    print(f"FINISHED TESTING THE `{dataset_name}` DATASET")


# MAIN ROUTINE
train(train_data, "training")
test(test_data, "testing")
