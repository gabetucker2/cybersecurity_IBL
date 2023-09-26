# SCRIPTS
import parameters
import functions_helper

def encode_chunk(inputs, output):

    data_to_encode = functions_helper.array_to_dictionary(inputs)
    data_to_encode[parameters.OUTPUT_NAME] = output

    parameters.memory.learn(data_to_encode, advance=1)
