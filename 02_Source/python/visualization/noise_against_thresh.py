import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import mpl_toolkits.mplot3d as a3
import matplotlib.pyplot as plt
import math

fig = plt.figure(figsize=(15, 12))
ax1 = fig.add_subplot(111)


data = np.array([[1,0.00005,506.5194805, 43.9030969, 140.8061938],
                 [2,0.0001,499.5514486, 71.61138861, 200.981019],
                 [3,0.0005,501.3826174, 152.012987, 353.4495504],
                 [4,0.001,500.2527473, 176.5724276, 409.4915085],
                 [5,0.005,499.4765235, 226.4875125, 486.984016],
                 [6,0.01,500.2697303, 240.038961, 495.6263736]])

plt.xticks(data[:,0],data[:,1])

for i in range(len(data)):
    ax1.scatter(data[i][0], data[i][2], color='#0465A9')
    ax1.scatter(data[i][0], data[i][3], color='#888888')
    ax1.scatter(data[i][0], data[i][4], color='#A41F22')



ax1.set_title("Feature Matching performance against Threshold")
ax1.set_xlabel("Threshold")
ax1.set_ylabel("# features")
ax1.grid()
plt.show()
