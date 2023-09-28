# LIBRARIES
import time
import warnings

# SCRIPTS
import parameters
import functions_train
import functions_test
import functions_preprocessing
import functions_helper

# MAIN

functions_preprocessing.preprocess_start()

time.sleep(0.1) # allows other scripts to initialize before starting
functions_preprocessing.preprocess_each_epoch()

if parameters.RUN_FEATURE_IMPORTANCE:
    print("RUNNING FEATURE IMPORTANCE ANALYSIS")

    warnings.filterwarnings("ignore") # ignore redundant warnings

    print(f"RUNNING `{parameters.ANALYSIS_FUNCTION.__name__}` FUNCTION")

    df_scores, df_col = parameters.ANALYSIS_FUNCTION() # call main function

    print(f"FINISHED `{parameters.ANALYSIS_FUNCTION.__name__}` FUNCTION")

    functions_helper.plotFeatures(df_scores, df_col)

    print(f"CREATED GRAPH AT `{functions_helper.getOutputName()}`")

if parameters.RUN_ACT_UP:
    print("RUNNING ACT-UP")

    accuracy_sum = 0

    for i in range(parameters.EPOCHS):
        
        functions_train.train(parameters.DATASET)

        time.sleep(parameters.READ_TRAIN_TIME)

        accuracy_sum += functions_test.test(parameters.DATASET, parameters.PROBABILITY_TEST, parameters.DECODE_FUNCTION)

        time.sleep(parameters.READ_TEST_TIME)

    print(f"MEAN {functions_helper.get_percent(accuracy_sum/parameters.EPOCHS)} ACCURACY ACROSS {parameters.EPOCHS} EPOCHS")
