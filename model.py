# SCRIPTS
import parameters
import functions_train
import functions_test
import functions_decode

##############################################################

# MAIN ROUTINE
functions_train.train(parameters.big_train_data, "big_train_data", 6)
functions_test.test(parameters.test_data, "test_data", 0.05, functions_decode.decode_chunk_blend)

##############################################################
