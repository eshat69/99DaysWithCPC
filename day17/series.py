"""num = [10, 20, 30, 40, 50]
sum=0
for x in num:
    sum = x*x
    print(x)
print(sum)
"""
n=int(input("Enter the last number: "))
sum1=0
for x in range(1,n+1,1) : #1+2+3+.....+n
    sum1=sum1+x
print(sum1)
sum2=0
for x in range(2,n+1,2): #odd series 2+4+..+n
    sum2=sum2+x
print(sum2)
sum3=0
for x in range (1,n+1,2): #even series 1+3+...+n
    sum3=sum3+x
print(sum3)
sum4=0
for x in range(4,n+1,4): #huul series 4+8+12+....+n
    sum4=sum4+x
print(sum4)
sum5=0
for x in range (1,n+1,1):  #squre series 1+4+9+16+...+n*n
    sum5=sum5+x*x
print(sum5)
sum6=1
for x in range (1,n+1): #factorial
    sum6=sum6*x
print(sum6)
a,b = 0,1
count=0
print("fibonacci  ", end="")
while count<n : #fibonacci series
    if count <= 1 :
        fibo = count
    else :
        fibo=a+b
        a=b
        b=fibo
    print(fibo, end=" ")
    count+=1
x, y = 2, 1  # Initial values for the Lucas series
count1 = 0
print("Lucas series:")
while count1 < n:
    if count1 == 0:
        lucas = x
    elif count1 == 1:
        lucas = y
    else:
        lucas = x + y
        x, y = y, lucas  # Update values
    print(lucas, end=" ")
    count1 += 1
