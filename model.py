# LIBRARIES
import pyactup
import pandas
import random
import numpy
import inspect

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
def get_percent(probability):
    return str(round(probability * 100, 2)) + '%'

def get_variable_name_as_string(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def encode_chunk(inputs, output):

    data_to_encode = {str(i): value for i, value in enumerate(inputs)}
    data_to_encode[OUTPUT_NAME] = output

    memory.learn(data_to_encode, advance=1)

def decode_chunk_best_blend(inputs):

    data_to_decode = {str(i): value for i, value in enumerate(inputs)}

    prediction, _ = memory.best_blend(OUTPUT_NAME, [data_to_decode])

    return prediction

def decode_chunk_retrieval_no_partial(inputs):

    data_to_decode = {str(i): value for i, value in enumerate(inputs)}

    prediction = (memory.retrieve(data_to_decode, partial=False) or {}).get(OUTPUT_NAME)

    return prediction

def decode_chunk_retrieval_partial(inputs):

    data_to_decode = {str(i): value for i, value in enumerate(inputs)}

    prediction = (memory.retrieve(data_to_decode, partial=True) or {}).get(OUTPUT_NAME)

    return prediction

# if threats_per_type is -1, fill to max capacity to save time
def train_populate_normal_half(original_dataset, dataset_name, threats_per_type):

    print(f"BEGINNING POPULATING THE `{dataset_name}` DATASET WITH PROPORTIONATE THREAT TYPES")

    out = 0

    if threats_per_type == -1:
        out = original_dataset
    else:
        
        # num_targets = {'val1': 0, 'val2': 0, "normal": 0}
        num_targets = {}
        for target_name in original_dataset[:, OUTPUT_COL_IDX]:
            num_targets[target_name] = 0

        # num_targets = {'val1': 4234, 'val2': 14, "normal": 532234}
        num_threat_types = len(num_targets.keys()) - 1 # other than normal
        total_trials = len(original_dataset[:, 0])
        for trial in range(total_trials):
            num_targets[original_dataset[trial, OUTPUT_COL_IDX]] += 1

        # selected_targets = {'val1': [0, 5, 9], 'val2': [2, 3, 6], 'normal': [0, 4, 6, 7, 9, 10]}
        def return_expected_threats(key, threats_per_type):
            out = threats_per_type
            if key == 'normal':
                out *= num_threat_types
            return out
        selected_targets = {
            key: random.sample(range(value), k = min(value, return_expected_threats(key, threats_per_type)))
            for key, value in num_targets.items()
        }

        # new_dataset = original_dataset WITHOUT ROWS NOT IN SELECTED_TARGETS
        new_dataset = numpy.empty((0, len(INPUT_COL_IDXS)+1))

        # num_targets = {'val1': 0, 'val2': 0, "normal": 0}
        for key, _ in num_targets.items():
            num_targets[key] = 0

        # populate new_dataset
        for trial in range(total_trials):
            training_output = original_dataset[trial, OUTPUT_COL_IDX]
            if num_targets[training_output] in selected_targets[training_output]:
                new_dataset = numpy.append(new_dataset, [original_dataset[trial, :]], axis=0)
            num_targets[training_output] += 1

        out = new_dataset

    print(f"FINISHED POPULATING THE `{dataset_name}` DATASET WITH PROPORTIONATE THREAT TYPES")

    return out

def train_learn(new_dataset, dataset_name):
    
    print(f"BEGINNING TRAINING THE `{dataset_name}` DATASET")
    
    total_trials = len(new_dataset[:, 0])
    
    for trial in range(total_trials):

        training_inputs = new_dataset[trial, INPUT_COL_IDXS]
        training_output = new_dataset[trial, OUTPUT_COL_IDX]

        encode_chunk(training_inputs, training_output)

    print(f"FINISHED TRAINING THE `{dataset_name}` DATASET")

def train(dataset, threats_per_type):
    
    dataset_name = get_variable_name_as_string(dataset)

    # get new dataset stochastically based on target counts
    new_dataset = train_populate_normal_half(dataset, dataset_name, threats_per_type)

    # train using new dataset from stochastic targets
    train_learn(new_dataset, dataset_name)

def test(dataset, trial_probability, decode_function):

    dataset_name = get_variable_name_as_string(dataset)

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
            testing_predicted = decode_function(testing_inputs)

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
train(big_train_data, -1)

# TEST
test(test_data, 0.01, decode_chunk_best_blend)
