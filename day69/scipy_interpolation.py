import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

x=np.linspace(0,5,10)
y=np.cos(x**2/3+4)
plt.scatter(x,y,c='red')

fun1=interp1d(x,y,kind='linear')
fun2=interp1d(x,y,kind='cubic')

xnew=np.linspace(0,4,10)
plt.plot(x,y,xnew,fun1(xnew),'--',xnew,fun2(xnew),'--')
plt.show()
