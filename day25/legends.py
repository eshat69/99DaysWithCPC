import matplotlib.pyplot as plt
x=["D1","D2","D3","D4","D5","D6","d7"]
y1=[30,20,10,34,67,45,90]
y2=[30,15,14,24,87,45,50]
y3=[30,20,70,4,7,45,50]
plt.figure(figsize=[5,3])
plt.plot(x,y1,label= "male")
plt.plot(x,y2,label= "female")
plt.plot(x,y3,label= "kids")
plt.legend()
plt.show()