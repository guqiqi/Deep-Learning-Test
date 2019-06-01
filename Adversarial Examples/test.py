import numpy as np

xs = [[1, 1, 1], [2, 3, 4], [2, 2, 2]]
xs = np.array(xs)
print(xs.shape)
print(xs.ndim)
print(len(xs))
tile = [len(xs)] + [1] * (xs.ndim + 1)
print(tile)
