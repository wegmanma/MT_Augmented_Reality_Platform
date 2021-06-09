import numpy as np

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(231, projection='3d')
ax4 = fig.add_subplot(234, projection='3d')
p = [[np.array([1., 1., 1.]), np.array([0., 0., 0.]), np.array([0., 0., 1.])], [
    np.array([0., 0., 0.]), np.array([-1., -1.,  1.]), np.array([0., 0., 1.])]]
fc = ['gold', 'crimson']
ax1.add_collection3d(Poly3DCollection(p, facecolors=fc, linewidths=1))
ax4.add_collection3d(Poly3DCollection(p, facecolors=fc, linewidths=1))
# Add 4th dimension to each point, ready for 4D transformation
for i in range(len(p)):
    for j in range(len(p[i])):
        p[i][j] = np.append(p[i][j], 1)

anglex = np.pi/2
angley = np.pi/2
anglez = np.pi/2

A_rot_x = np.array([[1, 0, 0, 0],
                    [0, np.cos(anglex), -np.sin(anglex), 0],
                    [0, np.sin(anglex), np.cos(anglex), 0],
                    [0, 0, 0, 1]])

A_rot_y = np.array([[np.cos(angley), 0, np.sin(angley), 0],
                    [0, 1, 0, 0],
                    [-np.sin(angley), 0, np.cos(angley), 0],
                    [0, 0, 0, 1]])

A_rot_z = np.array([[np.cos(anglez), np.sin(anglez), 0, 0],
                    [-np.sin(anglez), np.cos(anglez), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

A_trans = np.array([[1, 0, 0, 1],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

# Rotate only around Z
A = A_trans

p_rot1 = [[np.array([1., 1., 1.]), np.array([0., 0., 0.]), np.array([0., 0., 1.])], [
    np.array([0., 0., 0.]), np.array([-1., -1.,  1.]), np.array([0., 0., 1.])]]

for i in range(len(p)):
    for j in range(len(p[i])):
        temp = (np.matmul(A, p[i][j]))
        p_rot1[i][j] = temp[:-1]

ax2 = fig.add_subplot(232, projection='3d')
ax2.add_collection3d(Poly3DCollection(p_rot1, facecolors=fc, linewidths=1))

# Rotate around Z and then around X
A = A_rot_x @ A_rot_z

p_rot2 = [[np.array([1., 1., 1.]), np.array([0., 0., 0.]), np.array([0., 0., 1.])], [
    np.array([0., 0., 0.]), np.array([-1., -1.,  1.]), np.array([0., 0., 1.])]]
for i in range(len(p)):
    for j in range(len(p[i])):
        temp = (np.matmul(A, p[i][j]))
        p_rot2[i][j] = temp[:-1]

ax3 = fig.add_subplot(233, projection='3d')
ax3.add_collection3d(Poly3DCollection(p_rot2, facecolors=fc, linewidths=1))

# Rotate around X
A = A_rot_x

p_rot2 = [[np.array([1., 1., 1.]), np.array([0., 0., 0.]), np.array([0., 0., 1.])], [
    np.array([0., 0., 0.]), np.array([-1., -1.,  1.]), np.array([0., 0., 1.])]]
for i in range(len(p)):
    for j in range(len(p[i])):
        temp = (np.matmul(A, p[i][j]))
        p_rot2[i][j] = temp[:-1]

ax5 = fig.add_subplot(235, projection='3d')
ax5.add_collection3d(Poly3DCollection(p_rot2, facecolors=fc, linewidths=1))


# Rotate around X and then around Z
A = A_rot_z @ A_rot_x

p_rot2 = [[np.array([1., 1., 1.]), np.array([0., 0., 0.]), np.array([0., 0., 1.])], [
    np.array([0., 0., 0.]), np.array([-1., -1.,  1.]), np.array([0., 0., 1.])]]
for i in range(len(p)):
    for j in range(len(p[i])):
        temp = (np.matmul(A, p[i][j]))
        p_rot2[i][j] = temp[:-1]

ax6 = fig.add_subplot(236, projection='3d')
ax6.add_collection3d(Poly3DCollection(p_rot2, facecolors=fc, linewidths=1))

ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax3.set_xlim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)
ax4.set_xlim(-1.5, 1.5)
ax4.set_ylim(-1.5, 1.5)
ax5.set_xlim(-1.5, 1.5)
ax5.set_ylim(-1.5, 1.5)
ax6.set_xlim(-1.5, 1.5)
ax6.set_ylim(-1.5, 1.5)
plt.show()
