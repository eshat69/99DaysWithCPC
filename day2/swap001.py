w = int(input("enter w "))
x = int(input("enter x "))
y = int(input("enter y "))
z = int(input("enter z "))

temp = x
x = y
y = temp
print ("swap x,y",x,y)

w,z = z,w
print("swap w z",w,z)

