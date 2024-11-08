import timeit
import numpy as np
list_comp_time = timeit.timeit('[j**4 for j in range(1, 9)]', number=10)

numpy_time = timeit.timeit('np.arange(1, 9)**4', setup='import numpy as np', number=10)
print(f"List comprehension time: {list_comp_time}")
print(f"NumPy array operation time: {numpy_time}")

