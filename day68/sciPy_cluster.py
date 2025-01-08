import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.vq import kmeans, whiten

sns.set_style("darkgrid")
#create 50 datapoint in 2 cluster
pts = 50
#Data Generation:
a=np.random.multivariate_normal([0,0],[[4,1],[1,4]],size=pts)
b=np.random.multivariate_normal([30,10],[[10,2],[2,1]],size=pts)
features=np.concatenate((a,b))
w=whiten(features)
codebook,distortion = kmeans(w,2) #contains the centroids of the clusters
plt.scatter(w[:,0], w[:,1])
plt.scatter(codebook[:,0], codebook[:,1])
plt.show()