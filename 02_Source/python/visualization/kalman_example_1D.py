import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import mpl_toolkits.mplot3d as a3
import matplotlib.pyplot as plt
import math

# example motion, time series: n points
n = 1000

t = np.linspace(0, 25, n)
ns = 25/n

p = t.copy()
for i in range(len(p)):
    if t[i] >= 1 and t[i] <= 3:
        p[i] = (3/16)*t[i]**5+(-15/8)*t[i]**4+(55/8)*t[i]**3+(-45/4)*t[i]**2+(135/16)*t[i]+(-19/8)
    elif t[i] > 3 and t[i] < 5:
        p[i] = 1
    elif t[i] >= 5 and t[i] <= 8:
        p[i] = (-2/81)*t[i]**5+(65/81)*t[i]**4+(-830/81)*t[i]**3+(5200/81)*t[i]**2+(-16000/81)*t[i]+(19456/81)
    else:
        p[i] = 0.0
v = t.copy()
for i in range(len(v)):
    if i > 0:
        v[i] = (p[i]-p[i-1])/ns
    else:
        v[i] = 0
a = t.copy()
for i in range(len(a)):
    if i > 0:
        a[i] = (v[i]-v[i-1])/ns
    else:
        a[i] = 0


# add noise to a and v "measurement" and recalculate results from these inputs
an = a.copy()
vn = v.copy()
a_noise = np.random.normal(0, 0.01, n)
v_noise = np.random.normal(0, 0.1, n)
for i in range(len(an)):
    an[i] += a_noise[i]
    vn[i] += v_noise[i]
pvn = p.copy()
van = v.copy()
pan = p.copy()
for i in range(len(pvn)):
    if i > 0:
        pvn[i] = pvn[i-1]+vn[i]*ns
        van[i] = van[i-1]+an[i]*ns
    else:
        pvn[i] = 0
        van[i] = 0
for i in range(len(pvn)):
    if i > 0:
        pan[i] = pan[i-1]+van[i]*ns
    else:
        pan[i] = 0

# Kalman filter to combine these measurements

p_kalman = p.copy()
v_kalman = v.copy()
a_kalman = a.copy()

F = np.array([[1, ns, (ns**2)/2],[0, 1, ns],[0, 0, 1]])
Q = np.array([[(ns**4)/4, (ns**3)/2, (ns**2)/2],[(ns**3)/2, ns**2, ns],[(ns**2)/2, ns, 1]])
# initial condition, x_0 = 0
P_k = np.array(([[1, 0, 0],[0, 1, 0],[0, 0, 1]]))
P_k_k1 = np.array(([[1, 0, 0],[0, 1, 0],[0, 0, 1]]))
x_k = np.array([0, 0, 0])
x_k_k1 = np.array([0, 0, 0])
K = np.array([[0, 0],[0, 0],[0, 0]])
H = np.array([[0, 1, 0],[0, 0, 1]])

matrix1_kalman = p.copy()
matrix2_kalman = p.copy()
matrix3_kalman = p.copy()
p_manual = p.copy()
for i in range(n-1):
    # prediction
    x_k_k1 = np.matmul(F, x_k)
    P_k_k1 = np.matmul(F, np.matmul(P_k, np.transpose(F)))+Q
    #correction
    K = np.matmul(P_k_k1,np.matmul(np.transpose(H),(np.linalg.inv(np.matmul(H,np.matmul(P_k_k1,np.transpose(H)))+np.array([[300,0],[0,0.1]])))))
    tmp = np.matmul(K,(np.array([vn[i], an[i]])- np.matmul(H,x_k_k1)))
    tmp[0] = 0.0
    x_k = x_k_k1+np.matmul(K,(np.array([vn[i], an[i]])- np.matmul(H,x_k_k1)))
    P_k = np.matmul((np.identity(3)-np.matmul(K,H)),P_k_k1)
    p_kalman[i] = x_k[0]
    v_kalman[i] = x_k[1]
    a_kalman[i] = x_k[2]
    matrix1_kalman[i] = P_k[0][0]
    matrix2_kalman[i] = P_k[1][1]
    matrix3_kalman[i] = P_k[1][1]
    p_manual[i] = np.matmul(K,(np.array([vn[i], an[i]])- np.matmul(H,x_k_k1)))[0]



fig2 = plt.figure(figsize=(15, 12))
ax21 = fig2.add_subplot(111)
ax21.plot(t, p_manual, color='#0465A9')
# ax21.plot(t, matrix2_kalman, color='#544265')
# ax21.plot(t, matrix3_kalman, color='#A41F22')


fig = plt.figure(figsize=(15, 12))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)



ax1.plot(t, a, color='#0465A9')
ax1.plot(t, v, color='#544265')
ax1.plot(t, p, color='#A41F22')


ax2.plot(t, vn, color='#544265')
ax2.plot(t, pvn, color='#A41F22')



ax3.plot(t, an, color='#0465A9')
ax3.plot(t, van, color='#544265')
ax3.plot(t, pan, color='#A41F22')



ax4.plot(t, a_kalman, color='#0465A9')
ax4.plot(t, v_kalman, color='#544265')
ax4.plot(t, p_kalman, color='#A41F22')

# initial condition, x_0 = 0
P_k = np.array(([[1, 0, 0],[0, 1, 0],[0, 0, 1]]))
P_k_k1 = np.array(([[1, 0, 0],[0, 1, 0],[0, 0, 1]]))
x_k = np.array([0, 0, 0])
x_k_k1 = np.array([0, 0, 0])
K = np.array([[0],[0],[0]])
H = np.array([[0], [1], [0]])
for i in range(n-1):
    # prediction
    x_k_k1 = np.matmul(F, x_k)
    P_k_k1 = np.matmul(F, np.matmul(P_k, np.transpose(F)))+Q
    #correction
    K = np.matmul(P_k_k1,np.matmul(H,(np.linalg.inv(np.matmul(np.transpose(H),np.matmul(P_k_k1,H)+np.array([1]))))))
    x_k = x_k_k1+np.matmul(np.transpose(np.transpose(K)),(np.array([vn[i]])- np.matmul(np.transpose(H),np.transpose(x_k_k1))))
    P_k = np.matmul((np.identity(3)-np.matmul(K,np.transpose(H))),P_k_k1)
    p_kalman[i] = x_k[0]
    v_kalman[i] = x_k[1]
    a_kalman[i] = x_k[2]

# ax5.plot(t, v_kalman, color='#544265')
# ax5.plot(t, p_kalman, color='#A41F22')


# initial condition, x_0 = 0
P_k = np.array(([[1, 0, 0],[0, 1, 0],[0, 0, 1]]))
P_k_k1 = np.array(([[1, 0, 0],[0, 1, 0],[0, 0, 1]]))
x_k = np.array([0, 0, 0])
x_k_k1 = np.array([0, 0, 0])
K = np.array([[0],[0],[0]])
H = np.array([[0], [0], [1]])
for i in range(n-1):
    # prediction
    x_k_k1 = np.matmul(F, x_k)
    P_k_k1 = np.matmul(F, np.matmul(P_k, np.transpose(F)))+Q
    #correction
    K = np.matmul(P_k_k1,np.matmul(H,(np.linalg.inv(np.matmul(np.transpose(H),np.matmul(P_k_k1,H)+np.array([0.1]))))))
    x_k = x_k_k1+np.matmul(np.transpose(np.transpose(K)),(np.array([an[i]])- np.matmul(np.transpose(H),np.transpose(x_k_k1))))
    P_k = np.matmul((np.identity(3)-np.matmul(K,np.transpose(H))),P_k_k1)
    p_kalman[i] = x_k[0]
    v_kalman[i] = x_k[1]
    a_kalman[i] = x_k[2]

# ax6.plot(t, a_kalman, color='#0465A9')
# ax6.plot(t, v_kalman, color='#544265')
# ax6.plot(t, p_kalman, color='#A41F22')

ax1.set_ylim(-0.7, 1.5)
ax1.set_xlim(0, 20)
ax2.set_ylim(-0.7, 1.5)
ax2.set_xlim(0, 20)
ax3.set_ylim(-0.7, 1.5)
ax3.set_xlim(0, 20)
ax4.set_ylim(-0.7, 1.5)
ax4.set_xlim(0, 20)
# ax5.set_ylim(-0.2, 2.2)
# ax5.set_xlim(0, 20)
# ax6.set_ylim(-0.2, 2.2)
# ax6.set_xlim(0, 20)


ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax1.set_title('Simulation without noise')
ax2.set_title('Raw integration from noised velocity')
ax3.set_title('Raw integration from noised acceleration')
ax4.set_title('Kalman filter values from noised velocity and acceleration')
# ax5.grid()
# ax6.grid()
ax1.set_axisbelow(True)
ax2.set_axisbelow(True)
ax3.set_axisbelow(True)
ax4.set_axisbelow(True)
# ax5.set_axisbelow(True)
# ax6.set_axisbelow(True)

plt.show()
