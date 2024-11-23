import matplotlib.pyplot as plt
x = [2,4,6,9,8,6,1,4,5,7,8,9,5]
plt.stem(x,linefmt="--",markerfmt="D",bottom=0,orientation="vertical")
plt.plot(x)
plt.show()