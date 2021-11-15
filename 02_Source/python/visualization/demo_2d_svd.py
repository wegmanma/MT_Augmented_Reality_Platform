import numpy as np
import matplotlib.pyplot as plt

def cm_to_inch(value):
    return value/2.54

# define circle of single data points
n = 64
x = np.zeros((n,2))
for i in range(n):
    phi = i*(3.141592*2)/n
    x[i][0]=np.cos(phi)
    x[i][1]=np.sin(phi)

phi1 = 0.8
phi2 = (3.1415926*2)/128*53
U = np.array([[np.cos(phi1),-1.0*np.sin(phi1)],[np.sin(phi1),np.cos(phi1)]])
S = np.array([[1.4,0],[0,0.6]])
V = np.array([[np.cos(phi2),-1.0*np.sin(phi2)],[np.sin(phi2),np.cos(phi2)]])
Vt = np.transpose(V)

M = np.matmul(U,np.matmul(S,Vt))

Us, Ss, VTs = np.linalg.svd(M)


print("Results")
A_remake = (Us @ np.diag(Ss) @ VTs)
print(A_remake)

print(M)

fig = plt.figure(figsize=(cm_to_inch(40), cm_to_inch(25)))
ax1 = fig.add_subplot(221, aspect='equal')
ax2 = fig.add_subplot(222, aspect='equal')
ax3 = fig.add_subplot(223, aspect='equal')
ax4 = fig.add_subplot(224, aspect='equal')

y1 = x.copy()
for i in range(n):
    y1[i] = np.matmul(M,x[i])

y2 = x.copy()
for i in range(n):
    y2[i] = np.matmul(VTs,x[i])

y3 = x.copy()
for i in range(n):
    y3[i] = np.matmul(np.diag(Ss),y2[i])

y4 = x.copy()
for i in range(n):
    y4[i] = np.matmul(Us,y3[i])

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax1.set_axisbelow(True)
ax2.set_axisbelow(True)
ax3.set_axisbelow(True)
ax4.set_axisbelow(True)

for i in range(n):
    ax1.scatter(x[i][0],x[i][1],color='#0465A9')
    ax2.scatter(x[i][0],x[i][1],color='#0465A9')
    ax3.scatter(y2[i][0],y2[i][1],color='#0465A9')
    ax4.scatter(y3[i][0],y3[i][1],color='#0465A9')

for i in range(n):
    ax1.scatter(y1[i][0],y1[i][1],color='#A41F22')
    ax2.scatter(y2[i][0],y2[i][1],color='#A41F22')
    ax3.scatter(y3[i][0],y3[i][1],color='#A41F22')
    ax4.scatter(y4[i][0],y4[i][1],color='#A41F22')

ax1.arrow(0,0,x[0][0],x[0][1],color='#0465A9',width=0.02,length_includes_head=True)
ax1.arrow(0,0,y1[0][0],y1[0][1],color='#A41F22',width=0.02,length_includes_head=True)
ax2.arrow(0,0,x[0][0],x[0][1],color='#0465A9',width=0.02,length_includes_head=True)
ax2.arrow(0,0,y2[0][0],y2[0][1],color='#A41F22',width=0.02,length_includes_head=True)
ax3.arrow(0,0,y2[0][0],y2[0][1],color='#0465A9',width=0.02,length_includes_head=True)
ax3.arrow(0,0,y3[0][0],y3[0][1],color='#A41F22',width=0.02,length_includes_head=True)
ax4.arrow(0,0,y3[0][0],y3[0][1],color='#0465A9',width=0.02,length_includes_head=True)
ax4.arrow(0,0,y4[0][0],y4[0][1],color='#A41F22',width=0.02,length_includes_head=True)
ax1.scatter(x[0][0],x[0][1],color='#000000')
ax1.scatter(y1[0][0],y1[0][1],color='#000000')
ax1.scatter(0,0,color='#000000')
ax2.scatter(x[0][0],x[0][1],color='#000000')
ax2.scatter(y2[0][0],y2[0][1],color='#000000')
ax2.scatter(0,0,color='#000000')
ax3.scatter(y2[0][0],y2[0][1],color='#000000')
ax3.scatter(y3[0][0],y3[0][1],color='#000000')
ax3.scatter(0,0,color='#000000')
ax4.scatter(y3[0][0],y3[0][1],color='#000000')
ax4.scatter(y4[0][0],y4[0][1],color='#000000')
ax4.scatter(0,0,color='#000000')
ax1.set_ylim(-1.5, 1.5)
ax1.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.set_xlim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)
ax3.set_xlim(-1.5, 1.5)
ax4.set_ylim(-1.5, 1.5)
ax4.set_xlim(-1.5, 1.5)
ax1.set_title(r'$\vec{y}=M·\vec{x}=U·\Sigma·V^{T}·\vec{x}$')
ax2.set_title(r'$\vec{y}_{1}=V^{T}·\vec{x}$')
ax3.set_title(r'$\vec{y}_{2}=\Sigma·\vec{y}_{1}$')
ax4.set_title(r'$\vec{y}=U·\vec{y}_{2}$')


M_rec = np.matmul(np.transpose(x),y1)
print(M_rec)

Up, Sp, VTp = np.linalg.svd(M_rec)
S_det = np.identity(2)
S_det[1][1] = np.linalg.det(np.matmul(np.transpose(VTs),np.transpose(Us)))
R = np.matmul(np.transpose(VTs),np.matmul(S_det,np.transpose(Us)))
print(R)
S_det[1][1] = np.linalg.det(np.matmul(np.transpose(VTp),np.transpose(Up)))
Rp = np.matmul(np.transpose(VTp),np.matmul(S_det,np.transpose(Up)))
print(Rp)



fig.tight_layout()
plt.show()