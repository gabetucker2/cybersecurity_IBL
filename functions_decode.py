# SCRIPTS
import parameters
import functions_helper

def similarity(x, y):
    if isinstance(x, str) and isinstance(y, str):
        # Similarity computation for string values
        if x == y:
            return 1.0  # Exact match
        else:
            return 0.0  # No match
    elif isinstance(x, (float, int)) and isinstance(y, (float, int)):
        # Similarity computation for float or integer values
        if y < x:
            return f(y, x)
        return 1 - (y - x) / y
    else:
        return 0.0  # Default similarity for incompatible types

def decode_chunk_retrieval_strict(inputs):

    data_to_decode = functions_helper.array_to_dictionary(inputs)

    prediction = (parameters.memory.retrieve(data_to_decode, partial=False) or {}).get(parameters.OUTPUT_NAME)

    return prediction

def decode_chunk_retrieval_partial(inputs):

    data_to_decode = functions_helper.array_to_dictionary(inputs)

    # set the mismatch penalty for partial matching
    parameters.memory.mismatch = parameters.MISMATCH_PENALTY

    # call similarity functions for partial matching
    parameters.memory.similarity(
        attributes=list(data_to_decode.keys()),
        function=similarity,
        weight=parameters.SIMILARITY_WEIGHT
    )

    # Retrieve chunks with partial matching
    partial_chunks = parameters.memory.retrieve(data_to_decode, partial=True)

    # Check if any matching chunks were found
    prediction = partial_chunks.get(parameters.OUTPUT_NAME)

    return prediction

def decode_chunk_blend(inputs):

    data_to_decode = functions_helper.array_to_dictionary(inputs)

    # set the mismatch penalty for partial matching
    parameters.memory.mismatch = parameters.MISMATCH_PENALTY

    # call similarity functions for partial matching
    parameters.memory.similarity(
        attributes=list(data_to_decode.keys()),
        function=similarity,
        weight=parameters.SIMILARITY_WEIGHT
    )

    prediction, _ = parameters.memory.discrete_blend(parameters.OUTPUT_NAME, data_to_decode)

    return prediction
