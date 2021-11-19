import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import mpl_toolkits.mplot3d as a3
import matplotlib.pyplot as plt
import math

# example motion, time series: n points
n = 1000

t = np.linspace(0,10,n)
p = t.copy()
for i in range(len(p)):
    if t[i] >=1 and t[i]<=3:
        p[i]=-1*t[i]**3+6*t[i]**2-9*t[i]+4
    elif t[i] >3 and t[i]<5:
        p[i]=4
    elif t[i] >=5 and t[i]<=8:
        p[i]=(8/27)*t[i]**3-(52/9)*t[i]**2+(320/9)*t[i]-(1792/27)
    else:
        p[i] = 0.0
v =t.copy()
for i in range(len(v)):
    if i > 0:
        v[i] = p[i]-p[i-1]
    else:
        v[i] = 0
a =t.copy()
for i in range(len(a)):
    if i > 0:
        a[i] = v[i]-v[i-1]
    else:
        a[i] = 0



an = a.copy()
vn = v.copy()
a_noise = np.random.normal(0,0.0001,n)
v_noise = np.random.normal(0,0.01,n)
for i in range(len(an)):
    an[i] += a_noise[i]
    vn[i] += v_noise[i]
pvn = p.copy()
van = v.copy()
pan = p.copy()
for i in range(len(pvn)):
    if i > 0:
        pvn[i]=pvn[i-1]+vn[i]
        van[i]=van[i-1]+an[i]
    else:
        pvn[i] = 0
        van[i] = 0
for i in range(len(pvn)):
    if i > 0:
        pan[i]=pan[i-1]+van[i]
    else:
        pan[i] = 0


fig = plt.figure(figsize=(15, 12))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)



ax1.plot(t,p)
ax1.plot(t,v)
ax1.plot(t,a)

ax2.plot(t,pvn)
ax2.plot(t,vn)

ax3.plot(t,pan)
ax3.plot(t,van)
ax3.plot(t,an)

plt.show()