# LIBRARIES
import pyactup
import pandas
import random
import numpy

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

target_names = []

INPUT_COL_IDXS = [i for i in range(len(small_train_data[0, :])) if i != OUTPUT_COL_IDX]

# FUNCTIONS
def get_percent(probability):
    return str(round(probability * 100, 2)) + '%'

def get_target_idx(target_name):
    global target_names
    if target_name not in target_names:
        target_names.append(target_name)

def encode_chunk(inputs, output):

    data_to_encode = {str(i): value for i, value in enumerate(inputs)}
    data_to_encode[OUTPUT_NAME] = output

    memory.learn(data_to_encode, advance=1)

def decode_chunk(inputs):

    data_to_decode = {str(i): value for i, value in enumerate(inputs)}

    # prediction, _ = memory.best_blend(OUTPUT_NAME, [data_to_decode])
    prediction = (memory.retrieve(data_to_decode, partial=True) or {}).get(OUTPUT_NAME)

    return prediction

def train_populate(original_dataset, dataset_name, multiplier, num_targets):

    print(f"BEGINNING POPULATING THE `{dataset_name}` DATASET WITH PROPORTIONATE THREAT TYPES")
    
    # num_targets = {'val1': 3, 'val2': 3, "normal": 6}
    num_targets = {}
    for name in target_names:
        num_targets[name] = 0
    total_trials = len(original_dataset[:, 0])
    for trial in range(total_trials):
        num_targets[original_dataset[trial, OUTPUT_COL_IDX]] += 1
    num_targets["normal"] = 0
    num_targets["normal"] = sum(num_targets.values())

    # selected_targets = {'val1': [0, 5, 9], 'val2': [2, 3, 6], 'normal': [0, 4, 6, 7, 9, 10]}
    num_threat_types = len(target_names) - 1 # other than normal
    selected_targets = {
        key: random.sample(range(min(value, multiplier)), min(value, multiplier))
        for key, value in num_targets.items()
    }

    # new_dataset = original_dataset WITHOUT ROWS NOT IN SELECTED_TARGETS
    new_dataset = numpy.empty((sum(num_targets.values()), len(INPUT_COL_IDXS)+1))

    # num_targets = {'val1': 0, 'val2': 0, "normal": 0}
    for key, _ in num_targets.items():
        num_targets[key] = 0

    # populate new_dataset
    for trial in range(total_trials):
        training_output = original_dataset[trial, OUTPUT_COL_IDX]
        if num_targets[training_output] in selected_targets[training_output]:
            # add row to new_dataset if the current originaldataset row is in the chosen distribution
        num_targets[training_output] += 1

    print(f"FINISHED POPULATING THE `{dataset_name}` DATASET WITH PROPORTIONATE THREAT TYPES")

    return new_dataset

def train_learn(new_dataset, dataset_name):
    
    print(f"BEGINNING TRAINING THE `{dataset_name}` DATASET")
    
    total_trials = len(new_dataset[:, 0])
    
    for trial in range(total_trials):

        training_inputs = new_dataset[trial, INPUT_COL_IDXS]
        training_output = new_dataset[trial, OUTPUT_COL_IDX]

        encode_chunk(training_inputs, training_output)

    print(f"FINISHED TRAINING THE `{dataset_name}` DATASET")

def train(dataset, dataset_name, multiplier):
    
    # get new dataset stochastically based on target counts
    new_dataset = train_populate(dataset, dataset_name, multiplier)

    # train using new dataset from stochastic targets
    train_learn(new_dataset, dataset_name)

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
            print(f"Predicted: {testing_predicted}; actual: {testing_actual}; error? {testing_actual!=testing_predicted}")
            processed_trials += 1
    
    if processed_trials == 0:
        error_probability = 1
    else:
        error_probability = trial_errors / processed_trials
    
    print(f"Accuracy: {get_percent(error_probability)}")
    print(f"FINISHED TESTING THE `{dataset_name}` DATASET WITH {processed_trials} TRIALS ({get_percent(trial_probability)}) WITH {get_percent(MEMORY_NOISE)} NOISE")


# TRAIN
train(big_train_data, "big train", 1)

# TEST
test(test_data, "test", 0.01)
