import numpy as np
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(arr.shape)
arr2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
newarr = arr2.reshape(2, 3, 2)
print(newarr)
l = []
for i in range(1, 5):
    n = int(input("Enter a number: "))
    l.append(n)
print(np.array(l))
a = np.zeros(4)
print("zero", a)
b = np.ones(4)
print("one mat", b)
c = np.empty(4)
print("empty :", c)
d = np.eye(3)
print("identity", d)
r = np.arange(4)
print("range :", r)
lin = np.linspace(0, 10, num=5)
print("line space ", lin)
