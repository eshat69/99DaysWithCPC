def add(x,y):
    sum=x+y
    print(sum)
def sub(x,y):
    sub=x-y
    return sub
def empty():
    print("Empty function ")

add(1,4)
result=sub(2,1)
print(result)
empty()
def large(a,b):
    if a>b:
        return a
    else :
        return b
print(large(8,3))

sum1=(lambda a,b : a+b) (2,4)
print(sum1)
print(type(sum1))
print((lambda a,b : a*a + 2*a*b + b*b)  (2,3))
def sq(x):
    return x*x
num=[1,2,3,4,5]
result=list(map(sq,num))
print(result)
result2=list(filter(lambda x : x%2==0,num))
print(result2)
name=["hsf","sdui","wsdfhi","uisdf","ygfuy"]
print(list(zip(name,num)))