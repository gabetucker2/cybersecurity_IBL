# SCRIPTS
import parameters
import functions_encode

# FUNCTIONS

def train(dataset):

    dataset_name = dataset["name"]
    train = dataset["train_dict"] # use reduced training data
    
    print(f"BEGINNING TRAINING THE `{dataset_name}` DATASET")
    
    # encode chunks
    total_trials = train.shape[0]

    for trial in range(total_trials):

        training_inputs = train[trial, dataset["input_idxs"]]
        training_output = train[trial, dataset["output_idx"]]
        if parameters.BINARY:
            training_output = training_output == dataset["binary_target"]

        functions_encode.encode_chunk(training_inputs, training_output)

    print(f"FINISHED TRAINING THE `{dataset_name}` DATASET")
