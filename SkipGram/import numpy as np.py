import numpy as np

def sigmoidea(x):
    return 1 / (1 + np.exp(-x))

print((1/sigmoidea(-0) * sigmoidea(-0) * sigmoidea(0)))