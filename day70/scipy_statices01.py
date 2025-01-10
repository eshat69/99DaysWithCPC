from scipy.stats import describe, norm
from scipy import stats
import numpy as np
data = [0.2, -1.8, 0.34, -0.8, -0.2, 1.2, -1.0, 0.9]
description = describe(data)
for key, value in description._asdict().items():
    print(f"{key} : {value}")

dist = norm(0, 2)  # Normal distribution with mean=0 and standard deviation=2
arr = np.array([-1, 0.6, 1, 3, 2.5])
print(f"Cumulative distribution function: {dist.cdf(arr)}")
print(f"Percent point function (PPF) at 0.5: {dist.ppf(0.5)}")
print(f"Sampled distribution: {dist.rvs(size=5)}")
t_test = stats.ttest_1samp(data,1.1, alternative="less")
print(t_test)