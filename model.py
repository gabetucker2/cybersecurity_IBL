# LIBRARIES
import time

# SCRIPTS
import datasets
import functions_train
import functions_test
import functions_decode

# MAIN

dataset = datasets.dataset_USNW_NB15

time.sleep(0.1) # allows other scripts to initialize before starting

functions_train.train(dataset, 6)

time.sleep(1) # allows you to process training data

functions_test.test(dataset, 0.1, functions_decode.decode_chunk_blend)
