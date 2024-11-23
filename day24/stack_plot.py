import matplotlib.pyplot as plt
days=[1,2,3,4,5,6,7]
week1=[100,19,23,5,700,9,10]
week2=[500,600,400,200,400,300,700]
week3=[10,56,43,23,43,71,89]
plt.stackplot(days,week1,week2,week3,baseline="sym")
plt.show()
