print("enter 1 to 4")
x = int(input())
if x==1 :
    l=int(input("leanth ="))
    area=l**2
    print("squre = ",area)
elif x==2 :
    l=float(input("leanth = "))
    w=float(input("width "))
    area=l*w
    print("rectangle = ",area)
elif x==3 :
    r=float(input("redius ="))
    area=((3.1416)*(r**2))
    print("circle ",area)
elif x==4 :
    b=float(input("base m="))
    h=float(input("height ="))
    area=0.5*b*h
    print("triengle ",area)
else :
    print("x")