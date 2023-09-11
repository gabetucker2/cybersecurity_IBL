# LIBRARIES
import pandas

# DATASETS
dataset_NSL_KDD = {
    "name" : "NSL_KDD",
    "output_idx" : -2,
    "exclude_idxs" : [],
    "binary_target" : "normal",
    "train" : pandas.read_csv("../data/nsl_kdd/train.csv", header=None),
    "test" : pandas.read_csv("../data/nsl_kdd/test.csv", header=None)
}

dataset_USNW_NB15 = {
    "name" : "USNW_NB15",
    "output_idx" : -2,
    "exclude_idxs" : [
                        0, # remove idx column, since it's irrelevant
                        44 # remove binary classifier since we have our own system for this already
                    ],
    "binary_target" : "Normal",
    "train" : pandas.read_csv("../data/usnw_nb15/train.csv"),
    "test" : pandas.read_csv("../data/usnw_nb15/test.csv")
}

datasets = [dataset_NSL_KDD, dataset_USNW_NB15]

# PRE-PROCESSING
for dataset in datasets:
    # randomize order
    dataset["train"] = dataset["train"].sample(frac=1).reset_index(drop=True)
    dataset["test"] = dataset["test"].sample(frac=1).reset_index(drop=True)
    
    # fix the output_idx in case it's negative
    if dataset["output_idx"] < 0:
        dataset["output_idx"] = dataset["train"].shape[1] + dataset["output_idx"]
    
    # make new datasets for increased accessibility without having to index or column sample
    dataset["train_outputs"] = dataset["train"].iloc[:, [dataset["output_idx"]]]
    dataset["test_outputs"] = dataset["test"].iloc[:, [dataset["output_idx"]]]

    dataset["train_inputs"] = dataset["train"].drop(columns=dataset["train"].columns[dataset["exclude_idxs"]])
    dataset["test_inputs"] = dataset["test"].drop(columns=dataset["test"].columns[dataset["exclude_idxs"]])

    dataset["combined"] = pandas.concat([dataset["train"], dataset["test"]], ignore_index=True)
    dataset["combined_inputs"] = pandas.concat([dataset["train_inputs"], dataset["test_inputs"]], ignore_index=True)
    dataset["combined_outputs"] = pandas.concat([dataset["train_outputs"], dataset["test_outputs"]], ignore_index=True)
