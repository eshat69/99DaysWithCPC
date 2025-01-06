from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import Rbf
import numpy as np

xs = np.arange(10)
ys = 2*xs + 1
interp_func = interp1d(xs, ys)
newarr = interp_func(np.arange(2.1, 3, 0.1))
print(newarr)

xs2 = np.arange(10)
ys2 = xs2**2 + np.sin(xs) + 1
interp_func = UnivariateSpline(xs, ys)
newarr2 = interp_func(np.arange(2.1, 3, 0.1))
print(newarr2)

xs3 = np.arange(10)
ys3 = xs**2 + np.sin(xs) + 1
interp_func = Rbf(xs3, ys3)
newarr3 = interp_func(np.arange(2.1, 3, 0.1))
print(newarr3)

