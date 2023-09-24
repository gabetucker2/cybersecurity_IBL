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
