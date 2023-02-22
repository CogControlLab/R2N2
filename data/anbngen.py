"""
Generate dataset for anbn training
"""
import numpy as np
# import ipdb as pdb
import pickle

min_length = 1
max_length = 4

for min_length in [1 + i * 5 for i in range(8)]:
    max_length = min_length + 3

    data = []
    for length_a in range(min_length, max_length + 1):
        xs = np.zeros((length_a * 2 + 2, 3))
        # 0 for a, 1 for b, 2 for line break
        xs[:length_a, 0] = 1
        xs[length_a, 2] = 1
        xs[length_a + 1:-1, 1] = 1
        xs[-1, 2] = 1

        data.append(xs)

    pickle.dump(data, open(f"anbn_{min_length}-{max_length}.pkl", "wb"))
