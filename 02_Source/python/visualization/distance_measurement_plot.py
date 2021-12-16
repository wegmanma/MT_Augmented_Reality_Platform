import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import mpl_toolkits.mplot3d as a3
import matplotlib.pyplot as plt
import math

fig = plt.figure(figsize=(15, 12))
ax1 = fig.add_subplot(111)

data = np.array([[30, 0.3], [541, 0.4], [1040, 0.5], [1590, 0.6], [2162.191, 0.71], [
                2528.94, 0.8], [2983.776, 0.9], [3493.305, 1], [5630.247, 1.5], [7376.247, 1.99]])

trendline = np.array(
    [[0.000227*0+0.247532, 0], [0.000227*7500+0.247532, 7500]])

for i in range(len(data)):
    ax1.scatter(data[i][0], data[i][1], color='#0465A9')

ax1.plot([0, 7500], [0.000227*0+0.247532,
                     0.000227*7500+0.247532], color='#A41F22')

ax1.set_title("ToF Camera distance measurement")
ax1.set_xlabel("Camera Value")
ax1.set_ylabel("Distance [m]")
ax1.grid()
plt.show()
