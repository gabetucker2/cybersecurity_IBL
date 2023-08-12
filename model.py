# LIBRARIES
import time

# SCRIPTS
import datasets
import functions_train
import functions_test
import functions_decode

# MAIN

time.sleep(0.1) # allows other scripts to initialize before starting

functions_train.train(datasets.dataset_NSL, 6)
functions_test.test(datasets.dataset_NSL, 0.02, functions_decode.decode_chunk_blend)
