import numpy as np
import matplotlib.pyplot as plt

# define circle of single data points
n = 8
x = np.zeros((n,2))
for i in range(n):
    phi = i*(3.141592*2)/n
    x[i][0]=np.cos(phi)
    x[i][1]=np.sin(phi)

phi1 = 0.1
phi2 = -0.2
U = np.array([[np.cos(phi1),-1.0*np.sin(phi1)],[np.sin(phi1),np.cos(phi1)]])
S = np.array([[1.2,0],[0,1]])
V = np.array([[np.cos(phi2),-1.0*np.sin(phi2)],[np.sin(phi2),np.cos(phi2)]])
Vt = np.transpose(V)

M = np.matmul(U,np.matmul(S,Vt))

Us, Ds, VTs = np.linalg.svd(M)

print(Us)
print(Ds)
print(VTs)

print(M)