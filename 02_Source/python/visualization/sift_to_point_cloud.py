import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import math


def cm_to_inch(value):
    return value/2.54


def cubeface_to_poly(cubeface):
    points = [[0, 0], [0, 1], [1, 1], [1, 0]]
    for i in range(4):
        points[i][0] = cubeface[i][1]/cubeface[i][0]*-1.83
        points[i][1] = cubeface[i][2]/cubeface[i][0]*-1.83
    return points


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


# float x_div = coords_est[1] / coords_est[0];
# float x_tan = atan(x_div);
# point_new[idx].ransac_xpos_3d = (-128 * (x_tan - (WIDTH_ANGLE))) / (WIDTH_ANGLE);
# float y_div = coords_est[2] / coords_est[0];
# float y_tan = atan(y_div);
# point_new[idx].ransac_ypos_3d = (-102.5 * (y_tan - (HEIGHT_ANGLE))) / (HEIGHT_ANGLE);

R_mat = rotation_matrix([0.2, -0.5, 1], -1)

cubefaces_cent_orig = np.array([[[-1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0], [-1.0, 1.0, -1.0]],
                                [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0],
                                    [1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]],
                                [[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0]]])


cubefaces_cent = cubefaces_cent_orig.copy()
cubefaces_cent_rot = cubefaces_cent_orig.copy()
for i in range(3):
    for j in range(4):
        cubefaces_cent[i][j] = np.matmul(R_mat, cubefaces_cent_orig[i][j])

cubefaces = cubefaces_cent.copy()
cubefaces_rot = cubefaces.copy()
cubefaces_rot_prev = cubefaces.copy()
cubefaces_rot_prev2 = cubefaces.copy()

for i in range(3):
    for j in range(4):
        cubefaces[i][j][0] += 5

R_mat_2 = rotation_matrix([0, 1, 1], -0.1)
for i in range(3):
    for j in range(4):
        cubefaces_rot[i][j] = np.matmul(R_mat_2, cubefaces[i][j])

for i in range(3):
    for j in range(4):
        cubefaces_rot[i][j][1] += 0.1

R_mat_2 = rotation_matrix([0, 1, 1], 0.22)
for i in range(3):
    for j in range(4):
        cubefaces_rot_prev[i][j] = np.matmul(R_mat_2, cubefaces[i][j])

for i in range(3):
    for j in range(4):
        cubefaces_rot_prev[i][j][1] -= 0.0

R_mat_2 = rotation_matrix([0, 1, 1], 0.15)
for i in range(3):
    for j in range(4):
        cubefaces_rot_prev2[i][j] = np.matmul(R_mat_2, cubefaces_rot_prev[i][j])

for i in range(3):
    for j in range(4):
        cubefaces_rot_prev2[i][j][1] -= 0.0

points_0 = cubeface_to_poly(cubefaces[0])
points_1 = cubeface_to_poly(cubefaces[1])
points_2 = cubeface_to_poly(cubefaces[2])

points_0_rot = cubeface_to_poly(cubefaces_rot[0])
points_1_rot = cubeface_to_poly(cubefaces_rot[1])
points_2_rot = cubeface_to_poly(cubefaces_rot[2])

points_0_rot_prev = cubeface_to_poly(cubefaces_rot_prev[0])
points_1_rot_prev = cubeface_to_poly(cubefaces_rot_prev[1])
points_2_rot_prev = cubeface_to_poly(cubefaces_rot_prev[2])

points_0_rot_prev2 = cubeface_to_poly(cubefaces_rot_prev2[0])
points_1_rot_prev2 = cubeface_to_poly(cubefaces_rot_prev2[1])
points_2_rot_prev2 = cubeface_to_poly(cubefaces_rot_prev2[2])

fig = plt.figure(figsize=(cm_to_inch(40), cm_to_inch(30)))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
patches = []

polygon = Polygon(points_0, True, color='#0465A9')
patches.append(polygon)
polygon = Polygon(points_1, True, color='#0465A9')
patches.append(polygon)
polygon = Polygon(points_2, True, color='#0465A9')
patches.append(polygon)
polygon = Polygon(points_0_rot, True, color='#A41F22')
patches.append(polygon)
polygon = Polygon(points_1_rot, True, color='#A41F22')
patches.append(polygon)
polygon = Polygon(points_2_rot, True, color='#A41F22')
patches.append(polygon)

p = PatchCollection(patches, match_original=True, alpha=0.4)
ax1.add_collection(p)
for i in range(4):
    ax1.scatter(points_0[i][0], points_0[i][1], color='#0465A9')
    ax1.scatter(points_1[i][0], points_1[i][1], color='#0465A9')
    ax1.scatter(points_2[i][0], points_2[i][1], color='#0465A9')
    ax1.scatter(points_0_rot[i][0], points_0_rot[i][1], color='#A41F22')
    ax1.scatter(points_1_rot[i][0], points_1_rot[i][1], color='#A41F22')
    ax1.scatter(points_2_rot[i][0], points_2_rot[i][1], color='#A41F22')
ax1.set_ylim(-1, 1)
ax1.set_xlim(-1, 1)


patches = []
polygon = Polygon(points_0_rot_prev, True, color='#0465A9')
patches.append(polygon)
polygon = Polygon(points_1_rot_prev, True, color='#0465A9')
patches.append(polygon)
polygon = Polygon(points_2_rot_prev, True, color='#0465A9')
patches.append(polygon)
p = PatchCollection(patches, match_original=True, alpha=0.4)
ax2.add_collection(p)
for i in range(4):
    ax2.scatter(points_0_rot_prev[i][0], points_0_rot_prev[i][1], color='#0465A9')
    ax2.scatter(points_1_rot_prev[i][0], points_1_rot_prev[i][1], color='#0465A9')
    ax2.scatter(points_2_rot_prev[i][0], points_2_rot_prev[i][1], color='#0465A9')
ax2.set_ylim(-1, 1)
ax2.set_xlim(-1, 1)


patches = []
polygon = Polygon(points_0, True, color='#0465A9')
patches.append(polygon)
polygon = Polygon(points_1, True, color='#0465A9')
patches.append(polygon)
polygon = Polygon(points_2, True, color='#0465A9')
patches.append(polygon)
p = PatchCollection(patches, match_original=True, alpha=0.4)
ax3.add_collection(p)
for i in range(4):
    ax3.scatter(points_0[i][0], points_0[i][1], color='#0465A9')
    ax3.scatter(points_1[i][0], points_1[i][1], color='#0465A9')
    ax3.scatter(points_2[i][0], points_2[i][1], color='#0465A9')
ax3.set_ylim(-1, 1)
ax3.set_xlim(-1, 1)


patches = []
polygon = Polygon(points_0_rot, True, color='#0465A9')
patches.append(polygon)
polygon = Polygon(points_1_rot, True, color='#0465A9')
patches.append(polygon)
polygon = Polygon(points_2_rot, True, color='#0465A9')
patches.append(polygon)
p = PatchCollection(patches, match_original=True, alpha=0.4)
ax4.add_collection(p)
for i in range(4):
    ax4.scatter(points_0_rot[i][0], points_0_rot[i][1], color='#0465A9')
    ax4.scatter(points_1_rot[i][0], points_1_rot[i][1], color='#0465A9')
    ax4.scatter(points_2_rot[i][0], points_2_rot[i][1], color='#0465A9')
ax4.set_ylim(-1, 1)
ax4.set_xlim(-1, 1)

plt.show()
