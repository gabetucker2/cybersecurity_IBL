# LIBRARIES
import time

# SCRIPTS
import parameters
import functions_train
import functions_test
import functions_preprocessing
import functions_helper

# MAIN

functions_preprocessing.preprocess_start()

time.sleep(0.1) # allows other scripts to initialize before starting

accuracy_sum = 0

for i in range(parameters.EPOCHS):

    functions_preprocessing.preprocess_each_epoch()
    
    functions_train.train(parameters.DATASET)

    time.sleep(parameters.READ_TRAIN_TIME)

    accuracy_sum += functions_test.test(parameters.DATASET, parameters.PROBABILITY_TEST, parameters.DECODE_FUNCTION)

    time.sleep(parameters.READ_TEST_TIME)

print(f"MEAN {functions_helper.get_percent(accuracy_sum/parameters.EPOCHS)} ACCURACY ACROSS {parameters.EPOCHS} EPOCHS")
