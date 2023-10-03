# LIBRARIES
import numpy as np
import pandas as pd
import random
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# SCRIPTS
import parameters

# FUNCTIONS

def load_data():
    return parameters.DATASET["combined_inputs"]

def numeric_preprocessing(df):
    df_numeric = df.select_dtypes(include=[np.number])
    for feature in df_numeric.columns:
        if df_numeric[feature].max() > 10 * df_numeric[feature].median() and df_numeric[feature].max() > 10:
            df[feature] = np.where(df[feature] < df[feature].quantile(0.95), df[feature], df[feature].quantile(0.95))
        if df_numeric[feature].nunique() > 50:
            if df_numeric[feature].min() == 0:
                df[feature] = np.log(df[feature] + 1)
            else:
                df[feature] = np.log(df[feature])
    return df

def categorical_preprocessing(df):
    df_cat = df.select_dtypes(exclude=[np.number])
    for feature in df_cat.columns:
        if df_cat[feature].nunique() > 6:
            df[feature] = np.where(df[feature].isin(df[feature].value_counts().head().index), df[feature], '-')
    return df

def split_data(df):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.2, 
                                                        random_state = 0,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

def one_hot_encode_and_scale(X_train, X_test, df):
    df_cat = df.select_dtypes(exclude=[np.number])
    feature_names = list(X_train.columns)

    # Setting handle_unknown='ignore' for OneHotEncoder
    ct = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), [1,2,3])], 
        remainder='passthrough'
    )

    X_train = ct.fit_transform(X_train).toarray()
    
    if X_test is not None:  # only transform if X_test is provided
        X_test = ct.transform(X_test).toarray()

    # Since the encoder might ignore some categories in the test set, we should retrieve feature names from the encoder
    # to ensure consistency with the transformed columns.
    feature_names = ct.get_feature_names_out(input_features=feature_names)
    
    sc = StandardScaler()
    X_train[:, 18:] = sc.fit_transform(X_train[:, 18:])
    
    if X_test is not None:  # only scale if X_test is provided
        X_test[:, 18:] = sc.transform(X_test[:, 18:])

    return X_train, X_test, feature_names

# randomize order
def randomizeRows(dataset):
    dataset["train"] = dataset["train"].sample(frac=1).reset_index(drop=True)
    dataset["test"] = dataset["test"].sample(frac=1).reset_index(drop=True)

# fix the output_idx in case it's negative
def normalizeOutputIdx(dataset):
    if dataset["output_idx"] < 0:
        dataset["output_idx"] = dataset["train"].shape[1] + dataset["output_idx"]

# create input idxs
def createInputIdxs(dataset):
    dataset["input_idxs"] = [i for i in range(dataset["train"].shape[1]) if i != dataset["output_idx"] and i not in dataset["exclude_idxs"]]

# make new datasets for increased accessibility without having to index or column sample
def createSubDatasets(dataset):
    dataset["train_outputs"] = dataset["train"].iloc[:, [dataset["output_idx"]]]
    dataset["test_outputs"] = dataset["test"].iloc[:, [dataset["output_idx"]]]

    dataset["train_inputs"] = dataset["train"].drop(columns=dataset["train"].columns[dataset["exclude_idxs"]])
    dataset["test_inputs"] = dataset["test"].drop(columns=dataset["test"].columns[dataset["exclude_idxs"]])

    dataset["combined"] = pd.concat([dataset["train"], dataset["test"]], ignore_index=True)
    dataset["combined_inputs"] = pd.concat([dataset["train_inputs"], dataset["test_inputs"]], ignore_index=True)
    dataset["combined_outputs"] = pd.concat([dataset["train_outputs"], dataset["test_outputs"]], ignore_index=True)

def populate_all(dataset):

    dataset_name = dataset["name"]

    print(f"BEGINNING POPULATING THE `{dataset_name}` DATASET WITH ALL THREAT DATA")
    print(f"FINISHED POPULATING THE `{dataset_name}` DATASET WITH ALL THREAT DATA")

# 3-3-3-9 pattern based on arguments
def populate_normal_half(dataset):

    # init
    dataset_name = dataset["name"]
    train = dataset["train"].values # figure out where originaldataset is being updated such that values no longer exists
    output_idx = dataset["output_idx"]

    print(f"BEGINNING POPULATING THE `{dataset_name}` DATASET WITH PROPORTIONATE THREAT TYPES")

    # num_targets = {'val1': 0, 'val2': 0, "normal": 0}
    num_targets = {}
    for target_name in train[:, output_idx]:
        num_targets[target_name] = 0

    # num_targets = {'val1': 4234, 'val2': 14, "normal": 532234}
    num_threat_types = len(num_targets.keys()) - 1 # other than normal
    total_trials = train.shape[0]
    for trial in range(total_trials):
        num_targets[train[trial, output_idx]] += 1

    # selected_targets = {'val1': [0, 5, 9], 'val2': [2, 3, 6], 'normal': [0, 4, 6, 7, 9, 10]}
    def return_expected_threats(key):
        out = parameters.THREATS_PER_TYPE
        if key == dataset["binary_target"]:
            out *= num_threat_types
        return out
    selected_targets = {
        key: random.sample(range(value), k = min(value, return_expected_threats(key)))
        for key, value in num_targets.items()
    }

    # new_dataset = train WITHOUT ROWS NOT IN SELECTED_TARGETS
    dataset["train_dict"] = np.empty((0, train.shape[1]))

    # num_targets = {'val1': 0, 'val2': 0, "normal": 0}
    for key, _ in num_targets.items():
        num_targets[key] = 0

    # populate new_dataset
    for trial in range(total_trials):
        training_output = train[trial, output_idx]
        if num_targets[training_output] in selected_targets[training_output]:
            dataset["train_dict"] = np.append(dataset["train_dict"], [train[trial, :]], axis=0)
        num_targets[training_output] += 1

    print(f"FINISHED POPULATING THE `{dataset_name}` DATASET WITH PROPORTIONATE THREAT TYPES (`{parameters.THREATS_PER_TYPE}` threats per type)")

# reduce dataset training rows contingent on parameter settings
def selectTrainingRows(dataset):
    if parameters.THREATS_PER_TYPE == -1:
        populate_all(dataset)
    else:
        populate_normal_half(dataset)

def preprocess_start():

    normalizeOutputIdx(parameters.DATASET) # -2 in 5-long array becomes 3
    createInputIdxs(parameters.DATASET)
    createSubDatasets(parameters.DATASET) # train_outputs, test_inputs, combined, etc

def preprocess_each_epoch():

    randomizeRows(parameters.DATASET)
    selectTrainingRows(parameters.DATASET)
