import statistics
import math

n = int(input("no of el"))
list = []
counter = 1
for i in range(n):
    num = int(input("Enter value: "))
    list.append(num)
    counter += 1
mean = statistics.mean(list)
print("mean of the el : ", mean)
try:
    mode = statistics.mode(list)
    print("mode of the el : ", mode)
except statistics.StatisticsError:
    print("No unique mode found.")
median = statistics.median(list)
print("median of the el : ", median)
variance = sum((x - mean) ** 2 for x in list) / len(list)
print("variance of el : ", variance)
deviance = math.sqrt(variance)
print("standerd deviation of el : ", deviance)
