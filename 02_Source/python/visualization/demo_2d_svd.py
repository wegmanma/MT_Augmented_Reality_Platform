import numpy as np
import matplotlib.pyplot as plt

# define prototype point cloud
X0 = [1.5, 2, -1, -2, 0]
Y0 = [1, 0, -1, 2, 2]
C0 = ['#a31e22', '#7c3144', '#544264', '#2a5487', '#1465a8']

# define rotation and translation (in reality a-priori unknown)
phi = 0.4
rot = np.array([[np.cos(phi), (-1)*np.sin(phi)],
                [np.sin(phi), np.cos(phi)]])
tx = 0.7
ty = 0.1

# define rotated and translated point cloud and find centroids and center point clouds
X1 = [0, 0, 0, 0, 0]
Y1 = [0, 0, 0, 0, 0]
C1 = ['#c8787a', '#b1838f', '#998ea3', '#8098b7', '#72a3cb']
X0_cent = [0, 0, 0, 0, 0]
Y0_cent = [0, 0, 0, 0, 0]

cent0 = np.array([np.average(X0), np.average(Y0)])

for ii in range(len(X0)):
    vec = np.array([X0[ii], Y0[ii]])
    res = rot.dot(vec)
    X1[ii] = res[0]+tx
    Y1[ii] = res[1]+ty
    X0_cent[ii] = X0[ii]-cent0[0]
    Y0_cent[ii] = Y0[ii]-cent0[1]
X1_cent = [0, 0, 0, 0, 0]
Y1_cent = [0, 0, 0, 0, 0]
cent1 = np.array([np.average(X1), np.average(Y1)])
for ii in range(len(X0)):
    X1_cent[ii] = X1[ii]-cent1[0]
    Y1_cent[ii] = Y1[ii]-cent1[1]


# fill point clouds into DxN matrices and multiply to get a DxD (2x2 in this case)
S0 = np.zeros((2, 5))
S1 = np.zeros((2, 5))

for ii in range(len(X0)):
    S0[0, ii] = X0_cent[ii]
    S0[1, ii] = Y0_cent[ii]
    S1[0, ii] = X1_cent[ii]
    S1[1, ii] = Y1_cent[ii]
print(cent0)
print(S0)

print(cent1)
print(S1)

S = np.matmul(S0, np.transpose(S1))
print(S)
# Singular Value decomposition (SVD) on S
U, D, VT = np.linalg.svd(S)

print(U)
print(D)
print(VT)

# Create custom diagonal matrix according to ETH-Note
d_det = np.identity(2)
d_det[1, 1] = np.linalg.det(np.matmul(np.transpose(VT), np.transpose(U)))

# Rotation matrix shall be:
R_temp = np.matmul(np.transpose(VT), d_det)
R = np.matmul(R_temp, np.transpose(U))
print(R)
print(rot)  # QED

# translation between the point clouds is:
t = cent1-np.matmul(R, cent0)
print(t)  # QED

fig0, axes0 = plt.subplots(1, 1, figsize=(8, 6))
for ii in range(len(X0)):
    axes0.plot(X0[ii], Y0[ii], 'o', color=C0[ii])
    axes0.plot(X1[ii], Y1[ii], 'o', color=C1[ii])
axes0.plot(cent0[0], cent0[1], 'x', color="#000000")
axes0.plot(cent1[0], cent1[1], 'x', color="#777777")
axes0.set_title("Initial point clouds")
axes0.grid()

fig1, axes1 = plt.subplots(1, 1, figsize=(8, 6))
for ii in range(len(X0)):
    axes1.plot(X0_cent[ii], Y0_cent[ii], 'o', color=C0[ii])
    axes1.plot(X1_cent[ii], Y1_cent[ii], 'o', color=C1[ii])
axes1.plot(0, 0, 'x', color="#000000")
axes1.set_title("Centered point clouds")
axes1.grid()

plt.show()
