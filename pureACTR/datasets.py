# LIBRARIES
import pandas
import numpy

# DATASETS
dataset_NSL_KDD = {
    "name" : "NSL_KDD",
    "output_idx" : -2,
    "exclude_idxs" : [],
    "binary_target" : "normal",
    "train" : pandas.read_csv("../data/nsl_kdd/train.csv", header=None).values,
    "test" : pandas.read_csv("../data/nsl_kdd/test.csv", header=None).values
}

dataset_USNW_NB15 = {
    "name" : "USNW_NB15",
    "output_idx" : -2,
    "exclude_idxs" : [44], # data leakage, binary classifier
    "binary_target" : "Normal",
    "train" : pandas.read_csv("../data/usnw_nb15/train.csv").values,
    "test" : pandas.read_csv("../data/usnw_nb15/test.csv").values
}

datasets = [dataset_NSL_KDD, dataset_USNW_NB15]

# PROCEDURAL DATA
for dataset in datasets:
    # randomize row orders
    dataset["train"] = numpy.random.permutation(dataset["train"])
    dataset["test"] = numpy.random.permutation(dataset["test"])
    # normalize output idxs
    if dataset["output_idx"] < 0:
        dataset["output_idx"] = len(dataset["train"][0, :]) + dataset["output_idx"]
    # generate input_idxs
    dataset["input_idxs"] = [i for i in range(len(dataset["train"][0, :])) if i != dataset["output_idx"] and i not in dataset["exclude_idxs"]]
