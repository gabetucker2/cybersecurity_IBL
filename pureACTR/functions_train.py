# LIBRARIES
import numpy
import random

# SCRIPTS
import parameters
import functions_encode

# FUNCTIONS

def train_all(original_dataset):

    dataset_name = original_dataset["name"]

    print(f"BEGINNING POPULATING THE `{dataset_name}` DATASET WITH ALL THREAT DATA")
    print(f"FINISHED POPULATING THE `{dataset_name}` DATASET WITH ALL THREAT DATA")

    return original_dataset

# 3-3-3-9 pattern based on arguments
def train_populate_normal_half(original_dataset, threats_per_type):

    # init
    dataset_name = original_dataset["name"]
    data_train = original_dataset["train"]
    output_idx = original_dataset["output_idx"]

    print(f"BEGINNING POPULATING THE `{dataset_name}` DATASET WITH PROPORTIONATE THREAT TYPES")

    # num_targets = {'val1': 0, 'val2': 0, "normal": 0}
    num_targets = {}
    for target_name in data_train[:, output_idx]:
        num_targets[target_name] = 0

    # num_targets = {'val1': 4234, 'val2': 14, "normal": 532234}
    num_threat_types = len(num_targets.keys()) - 1 # other than normal
    total_trials = len(data_train[:, 0])
    for trial in range(total_trials):
        num_targets[data_train[trial, output_idx]] += 1

    # selected_targets = {'val1': [0, 5, 9], 'val2': [2, 3, 6], 'normal': [0, 4, 6, 7, 9, 10]}
    def return_expected_threats(key, threats_per_type):
        out = threats_per_type
        if key == original_dataset["binary_target"]:
            out *= num_threat_types
        return out
    selected_targets = {
        key: random.sample(range(value), k = min(value, return_expected_threats(key, threats_per_type)))
        for key, value in num_targets.items()
    }

    # new_dataset = data_train WITHOUT ROWS NOT IN SELECTED_TARGETS
    new_dataset = original_dataset
    new_dataset["train"] = numpy.empty((0, original_dataset["train"].shape[1]))

    # num_targets = {'val1': 0, 'val2': 0, "normal": 0}
    for key, _ in num_targets.items():
        num_targets[key] = 0

    # populate new_dataset
    for trial in range(total_trials):
        training_output = data_train[trial, output_idx]
        if num_targets[training_output] in selected_targets[training_output]:
            new_dataset["train"] = numpy.append(new_dataset["train"], [data_train[trial, :]], axis=0)
        num_targets[training_output] += 1

    print(f"FINISHED POPULATING THE `{dataset_name}` DATASET WITH PROPORTIONATE THREAT TYPES (`{threats_per_type}` threats per type)")

    return new_dataset

def train_learn(new_dataset):

    dataset_name = new_dataset["name"]
    
    print(f"BEGINNING TRAINING THE `{dataset_name}` DATASET")
    
    # encode chunks
    total_trials = len(new_dataset["train"][:, 0])

    for trial in range(total_trials):

        training_inputs = new_dataset["train"][trial, new_dataset["input_idxs"]]
        training_output = new_dataset["train"][trial, new_dataset["output_idx"]]
        if parameters.BINARY:
            training_output = training_output == new_dataset["binary_target"]

        functions_encode.encode_chunk(training_inputs, training_output)

    print(f"FINISHED TRAINING THE `{dataset_name}` DATASET")

def train(dataset, threats_per_type):

    # get new dataset stochastically based on target counts
    new_dataset = 0
    if threats_per_type == -1:
        new_dataset = train_all(dataset)
    else:
        new_dataset = train_populate_normal_half(dataset, threats_per_type)

    # train using new dataset from stochastic targets
    train_learn(new_dataset)
    