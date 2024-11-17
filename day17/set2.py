num1={1,2,3,4,5}
num2=set([4,5,6,7,8])
print(num1)
print(num2)
print(num1 | num2) #union
print(num1 & num2) #intersection
print(num1 ^ num2)  # symmetric difference (common items vanish)
print(num1 - num2) #1-2 1 er common iteam bad
print(num2 - num1) #2-1 2 er common irem bad