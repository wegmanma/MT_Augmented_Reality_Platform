import numpy as np
from numpy import linalg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt


def cm_to_inch(value):
    return value/2.54


fig = plt.figure(figsize=(cm_to_inch(40), cm_to_inch(30)))

ax1 = fig.add_subplot(231, projection='3d')
ax2 = fig.add_subplot(232, projection='3d')
ax3 = fig.add_subplot(233, projection='3d')
ax4 = fig.add_subplot(234, projection='3d')
ax5 = fig.add_subplot(235, projection='3d')
ax6 = fig.add_subplot(236, projection='3d')

ax1.plot([0, 0], [0, 0], [0, 1.5], '#0465A9')
ax1.plot([-1.5, 1.5], [0, 0], [0, 0], '#0465A9')
ax1.plot([0, 0], [-1.5, 1.5], [0, 0], '#0465A9')
ax2.plot([0, 0], [0, 0], [0, 1.5], '#0465A9')
ax2.plot([-1.5, 1.5], [0, 0], [0, 0], '#0465A9')
ax2.plot([0, 0], [-1.5, 1.5], [0, 0], '#0465A9')
ax3.plot([0, 0], [0, 0], [0, 1.5], '#0465A9')
ax3.plot([-1.5, 1.5], [0, 0], [0, 0], '#0465A9')
ax3.plot([0, 0], [-1.5, 1.5], [0, 0], '#0465A9')
ax4.plot([0, 0], [0, 0], [0, 1.5], '#0465A9')
ax4.plot([-1.5, 1.5], [0, 0], [0, 0], '#0465A9')
ax4.plot([0, 0], [-1.5, 1.5], [0, 0], '#0465A9')
ax5.plot([0, 0], [0, 0], [0, 1.5], '#0465A9')
ax5.plot([-1.5, 1.5], [0, 0], [0, 0], '#0465A9')
ax5.plot([0, 0], [-1.5, 1.5], [0, 0], '#0465A9')
ax6.plot([0, 0], [0, 0], [0, 1.5], '#0465A9')
ax6.plot([-1.5, 1.5], [0, 0], [0, 0], '#0465A9')
ax6.plot([0, 0], [-1.5, 1.5], [0, 0], '#0465A9')

p = [[np.array([1., 1., 1.]), np.array([0., 0., 0.]), np.array([0., 0., 1.])], [
    np.array([0., 0., 0.]), np.array([-1., -1.,  1.]), np.array([0., 0., 1.])]]
fc = ['#544265', '#A41F22']
ax1.add_collection3d(Poly3DCollection(p, facecolors=fc, linewidths=1))
ax4.add_collection3d(Poly3DCollection(p, facecolors=fc, linewidths=1))
# Add 4th dimension to each point, ready for 4D transformation
for i in range(len(p)):
    for j in range(len(p[i])):
        p[i][j] = np.append(p[i][j], 1)

anglex = np.pi/4
angley = np.pi/4
anglez = np.pi/4

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

print("============X============")
A = A_rot_x
A3 = A[:-1][:, :-1]
w, v = linalg.eig(A3)
print(w[2])
print(v[:, 2])
print("============Y============")
A = A_rot_y
A3 = A[:-1][:, :-1]
w, v = linalg.eig(A3)
print(w[2])
print(v[:, 2])
print("============Z============")
A = A_rot_z
A3 = A[:-1][:, :-1]
w, v = linalg.eig(A3)
print(w[2])
print(v[:, 2])
print("==========XZ============")
A = A_rot_x @ A_rot_z
A3 = A[:-1][:, :-1]
w, v = linalg.eig(A3)
print(w[2])

print("What Gyro would give us...")
print(v[:, 2])
print(np.imag(np.log(w[0])/j))

# Unit Quaternions! ("Normalized Quaternions")

# Rotate only around Z
A = A_rot_z

p_rot1 = [[np.array([1., 1., 1.]), np.array([0., 0., 0.]), np.array([0., 0., 1.])], [
    np.array([0., 0., 0.]), np.array([-1., -1.,  1.]), np.array([0., 0., 1.])]]

for i in range(len(p)):
    for j in range(len(p[i])):
        temp = (np.matmul(A, p[i][j]))
        p_rot1[i][j] = temp[:-1]


ax2.add_collection3d(Poly3DCollection(p_rot1, facecolors=fc, linewidths=1))

# Rotate around Z and then around X
A = A_rot_x @ A_rot_z

p_rot2 = [[np.array([1., 1., 1.]), np.array([0., 0., 0.]), np.array([0., 0., 1.])], [
    np.array([0., 0., 0.]), np.array([-1., -1.,  1.]), np.array([0., 0., 1.])]]
for i in range(len(p)):
    for j in range(len(p[i])):
        temp = (np.matmul(A, p[i][j]))
        p_rot2[i][j] = temp[:-1]


ax3.add_collection3d(Poly3DCollection(p_rot2, facecolors=fc, linewidths=1))

# Rotate around X
A = A_rot_x

p_rot2 = [[np.array([1., 1., 1.]), np.array([0., 0., 0.]), np.array([0., 0., 1.])], [
    np.array([0., 0., 0.]), np.array([-1., -1.,  1.]), np.array([0., 0., 1.])]]
for i in range(len(p)):
    for j in range(len(p[i])):
        temp = (np.matmul(A, p[i][j]))
        p_rot2[i][j] = temp[:-1]


ax5.add_collection3d(Poly3DCollection(p_rot2, facecolors=fc, linewidths=1))


# Rotate around X and then around Z
A = A_rot_z @ A_rot_x

p_rot2 = [[np.array([1., 1., 1.]), np.array([0., 0., 0.]), np.array([0., 0., 1.])], [
    np.array([0., 0., 0.]), np.array([-1., -1.,  1.]), np.array([0., 0., 1.])]]
for i in range(len(p)):
    for j in range(len(p[i])):
        temp = (np.matmul(A, p[i][j]))
        p_rot2[i][j] = temp[:-1]


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
ax1.set_title('Original', fontsize=16)
ax2.set_title('Rotate around Z', fontsize=16)
ax3.set_title('Rotate around Z and then X', fontsize=16)
ax4.set_title('Original', fontsize=16)
ax5.set_title('Rotate around X', fontsize=16)
ax6.set_title('Rotate around X and then Z', fontsize=16)
plt.tight_layout()
plt.show()
