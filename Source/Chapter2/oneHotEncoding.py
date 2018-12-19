import numpy as np


def get_one_hot(input_vector):
    result = []
    max_val = max(input_vector)
    for i in input_vector:
        new_val = np.zeros(max_val)
        new_val.itemset(i-1, 1)
        result.append(new_val)
    return result


res = get_one_hot([1, 5, 2, 4, 3])
print(res)
