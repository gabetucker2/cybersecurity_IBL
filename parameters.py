# LIBRARIES
import pyactup

# STATIC
OUTPUT_NAME = "threat category"
MEMORY_NOISE = 0.1
MISMATCH_PENALTY = 0.5
SIMILARITY_WEIGHT = 3

# PROCEDURAL
memory = pyactup.Memory(noise=MEMORY_NOISE)
