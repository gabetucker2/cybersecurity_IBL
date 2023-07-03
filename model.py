# LIBRARIES
import pyactup
import pandas
import random

# PARAMETERS
SMALL_TRAINING_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/Small%20Training%20Set.csv"
BIG_TRAINING_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.csv"
TESTING_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.csv"
OUTPUT_COL_IDX = 41
OUTPUT_NAME = "threat category"

MEMORY_NOISE = 0.1

# VARIABLES
small_train_data = pandas.read_csv(SMALL_TRAINING_URL, header=None).values
big_train_data = pandas.read_csv(BIG_TRAINING_URL, header=None).values
test_data = pandas.read_csv(TESTING_URL, header=None).values
memory = pyactup.Memory(noise=MEMORY_NOISE)

INPUT_COL_IDXS = [i for i in range(len(small_train_data[0, :])) if i != OUTPUT_COL_IDX]

# FUNCTIONS
def encode_chunk(inputs, output):

    data_to_encode = {}

    for i in range(len(inputs)):
        data_to_encode[f'{i}'] = inputs[i]
    
    data_to_encode[OUTPUT_NAME] = output

    memory.learn(data_to_encode)
    memory.advance()

def decode_chunk(inputs):

    data_to_decode = {}

    for i in range(len(inputs)):
        data_to_decode[f'{i}'] = inputs[i]

    return (memory.retrieve(data_to_decode) or {}).get(OUTPUT_NAME)

def train(dataset, dataset_name, trial_probability):

    print(f"BEGINNING TRAINING THE `{dataset_name}` DATASET")

    processed_trials = 0
    total_trials = len(dataset[:, 0])
    print_interval = [(i+1) * (total_trials // 10) for i in range(10)]

    for trial in range(total_trials):

        if trial in print_interval:
            print(f"{(print_interval.index(trial)+1)*10}% trained")

        if trial_probability > random.random():

            training_inputs = dataset[trial, INPUT_COL_IDXS]
            training_output = dataset[trial, OUTPUT_COL_IDX]

            encode_chunk(training_inputs, training_output)

            processed_trials += 1
    
    print(f"FINISHED TRAINING THE `{dataset_name}` DATASET WITH {processed_trials} TRIALS")

def test(dataset, dataset_name, trial_probability):

    print(f"BEGINNING TESTING THE `{dataset_name}` DATASET")

    trial_errors = 0
    processed_trials = 0
    total_trials = len(dataset[:, 0])
    print_interval = [(i+1) * (total_trials // 10) for i in range(10)]

    for trial in range(total_trials):

        if trial in print_interval:
            print(f"{(print_interval.index(trial)+1)*10}% tested")

        if trial_probability > random.random():

            testing_inputs = dataset[trial, INPUT_COL_IDXS]
            testing_actual = dataset[trial, OUTPUT_COL_IDX]
            testing_predicted = decode_chunk(testing_inputs)

            trial_errors += testing_actual == testing_predicted
            processed_trials += 1
    
    if processed_trials == 0:
        error_probability = 1
    else:
        error_probability = trial_errors / processed_trials
    
    print(f"Accuracy: {(error_probability * 100)}%")
    print(f"FINISHED TESTING THE `{dataset_name}` DATASET WITH {processed_trials} TRIALS")


# TRAIN
train(big_train_data, "big train", 1)

# TEST
test(test_data, "test", 0.05)
