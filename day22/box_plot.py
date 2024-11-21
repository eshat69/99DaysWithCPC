import matplotlib.pyplot as plt
l=[1,3,4,7,12,2,8,9,24]
l2=[2,3,4,6,3,5,7,3,6]
l3=[2,5,12,3,23,12,3,7,9,10,15,19]
plot_value= [l,l2,l3]
plt.boxplot(plot_value)
plt.show()