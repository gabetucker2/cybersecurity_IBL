# FUNCTIONS
def get_percent(probability):
    return str(round(probability * 100, 2)) + '%'

def array_to_dictionary(arr):
    return {str(i): value for i, value in enumerate(arr)}
