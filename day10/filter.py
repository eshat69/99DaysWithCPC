import numpy as np
arr = np.array([10,4,29,6])
x = [True, False, True, False]
n = arr[x]
print(n)
arr = np.array([41, 42, 43, 44])
filter_arr = []
for element in arr:
  if element > 42:
    filter_arr.append(True)
  else:
    filter_arr.append(False)
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)

arr = np.array([34,5,345,23,23,25,667,345,23,])
filter_arr = arr > 42
arr2 = arr[filter_arr]
print(filter_arr)
print(arr)
filter_arr = arr % 2 == 0
print(filter_arr)
