n=int(input())
temp=n
reverse=0
while n>0 :
    digit=n%10
    reverse=reverse*10+digit
    n=n//10
print(reverse)
if temp==reverse :
    print("palindrom")
else :
    print("not palindrom")