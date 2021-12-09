import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import mpl_toolkits.mplot3d as a3
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

fig = plt.figure()
fig2 = plt.figure()
ax1 = fig2.add_subplot(111, projection='3d')
ax2 = fig.add_subplot(221, projection='3d')
ax3 = fig.add_subplot(222)
ax5 = fig.add_subplot(223, projection='3d')
ax6 = fig.add_subplot(224)
fov = np.array([[1.0, -1.47398708, -0.839102161, 1.], [1.0, 1.47398708, -0.839102161, 1.],
                [1.0, 1.47398708, 0.839102161, 1.], [1.0, -1.47398708, 0.839102161, 1.]],
               dtype=object)

ax1.quiver(0, 0, 0, 1, 0, 0, color='gray', alpha=.8, lw=1)
ax1.quiver(0, 0, 0, 0, 1, 0, color='gray', alpha=.8, lw=1)
ax1.quiver(0, 0, 0, 0, 0, 1, color='gray', alpha=.8, lw=1)
ax5.quiver(0, 0, 0, 1, 0, 0, color='gray', alpha=.8, lw=1)
ax5.quiver(0, 0, 0, 0, 1, 0, color='gray', alpha=.8, lw=1)
ax5.quiver(0, 0, 0, 0, 0, 1, color='gray', alpha=.8, lw=1)
ax2.quiver(0, 0, 0, 1, 0, 0, color='#A41F22', alpha=.8, lw=1)
ax2.quiver(0, 0, 0, 0, 0, 1, color='#A41F22', alpha=.8, lw=1)
ax2.scatter(0, 0, 0, color='#A41F22')
ax2.quiver(0, 0, 0, 0, 1, 0, color='gray', alpha=.8, lw=1)


vertices = np.array([[0.0, 1.0, 0.5625, 1.], [0.0, -1.0, 0.5625, 1.],
                     [0.0, -1.0, -0.5625, 1.], [0.0, 1.0, -0.5625, 1.]],
                    dtype=object)

ubo_model = np.array([[1., 0., 0., 2.], [0., 1., 0., 0.],
                      [0., 0., 1., 0.], [0., 0., 0., 1.]])

ubo_view = np.array([[0., -1., 0., 0.], [0., 0., 1., 0.],
                     [-1., 0., 0., 0.], [0., 0., 0., 1.]])

ubo_proj = np.array([[0.678432, 0., 0., 0.], [0., -1.19175, 0., 0.],
                     [0., 0., -1.0202, -0.20202], [0., 0., -1., 0.]])

A = np.matmul(ubo_proj, np.matmul(ubo_view, ubo_model))

print(A)
clip_space = np.copy(vertices)
clip_space[0] = np.matmul(A, vertices[0])

world_space = np.copy(vertices)
world_space[0] = np.matmul(ubo_model, vertices[0])


for i in range(4):
    clip_space[i] = np.matmul(A, vertices[i])
    world_space[i] = np.matmul(ubo_model, vertices[i])
for i in range(4):
    color = '#0465A9'
    if (i == 0):
        color = '#8dc048'
        print("dot 1")
        print(vertices[0])
        print(clip_space[0])
        print(clip_space[0][0]/clip_space[0][3])
        print(clip_space[0][1]/clip_space[0][3])
    ax2.plot([0, fov[i][0]],
             [0, fov[i][1]],
             zs=[0, fov[i][2]], color='#000000', alpha=.8, lw=1)
    ax1.scatter(vertices[i][0], vertices[i][1], vertices[i]
                [2], color=color, alpha=.8, lw=1)
    ax2.scatter(world_space[i][0], world_space[i][1], world_space[i]
                [2], color=color, alpha=.8, lw=1)
    ax3.scatter(clip_space[i][0]/clip_space[i][3],
                clip_space[i][1]/clip_space[i][3], color=color)
    ax1.plot([vertices[i][0], vertices[(i+1) % 4][0]],
             [vertices[i][1], vertices[(i+1) % 4][1]],
             zs=[vertices[i][2], vertices[(i+1) % 4][2]], color='#000000', alpha=.8, lw=1)
    ax2.plot([world_space[i][0], world_space[(i+1) % 4][0]],
             [world_space[i][1], world_space[(i+1) % 4][1]],
             zs=[world_space[i][2], world_space[(i+1) % 4][2]], color='#000000', alpha=.8, lw=1)
    ax3.plot([clip_space[i][0]/clip_space[i][3], clip_space[(i+1) % 4][0]/clip_space[i][3]],
             [clip_space[i][1]/clip_space[i][3], clip_space[(i+1) % 4][1]/clip_space[i][3]], color='#000000', alpha=.8, lw=1)
# 
rot = np.array([[0.906308, -0.298836, 0.298836, 0], [0.298836, 0.953154, 0.0468461, 0],
                [-0.298836, 0.0468461, 0.953154, 0], [0, 0, 0, 1]])
total_matrix = np.array([
    [0.180446, -0.653538, -0.0244536, 0.385345],
    [-0.316565, -0.0444475, -1.14808, 0.51495],
    [0.94538, 0.271172, -0.271172, 1.95991],
    [0.92666, 0.265802, -0.265802, 2.11912]])
fov_rot = np.copy(fov)
clip_space_rot = np.copy(clip_space)
for i in range(4):
    clip_space_rot[i] = np.matmul(total_matrix, vertices[i])
    fov_rot[i] = np.matmul(rot, fov[i])+[0,0,1,0]
for i in range(4):
    color = '#0465A9'
    if (i == 0):
        color = '#8dc048'
        print("dot 1 rot")
        print(vertices[0])
        print(clip_space_rot[0])
        print(clip_space_rot[0][0]/clip_space_rot[0][3])
        print(clip_space_rot[0][1]/clip_space_rot[0][3])
    ax5.plot([0, fov_rot[i][0]],
             [0, fov_rot[i][1]],
             zs=[1, fov_rot[i][2]], color='#000000', alpha=.8, lw=1)
    ax5.scatter(world_space[i][0], world_space[i][1], world_space[i]
                [2], color=color, alpha=.8, lw=1)
    ax5.plot([world_space[i][0], world_space[(i+1) % 4][0]],
             [world_space[i][1], world_space[(i+1) % 4][1]],
             zs=[world_space[i][2], world_space[(i+1) % 4][2]], color='#000000', alpha=.8, lw=1)
    ax6.scatter(clip_space_rot[i][0]/clip_space_rot[i][3],
                clip_space_rot[i][1]/clip_space_rot[i][3], color=color)
    ax6.plot([clip_space_rot[i][0]/clip_space_rot[i][3], clip_space_rot[(i+1) % 4][0]/clip_space_rot[(i+1) % 4][3]],
             [clip_space_rot[i][1]/clip_space_rot[i][3], clip_space_rot[(i+1) % 4][1]/clip_space_rot[(i+1) % 4][3]], color='#000000', alpha=.8, lw=1)

ax5.scatter(0, 0, 1, color='#A41F22')
ax5.quiver(0, 0, 1, 1.205975, 0.345682, 0.654318-1, color='#A41F22', alpha=.8, lw=1)
ax5.quiver(0, 0, 1, 0.298836, 0.0468461, 0.953154, color='#A41F22', alpha=.8, lw=1)
ax3.set_ylim(1, -1)
ax3.set_xlim(-1, 1)
ax6.set_ylim(1, -1)
ax6.set_xlim(-1, 1)


ax1.set_ylim(-1.2, 1.2)
ax1.set_xlim(-1.2, 1.2)
ax1.set_zlim(-1.2, 1.2)
ax1.set_title("Local Space")
ax2.set_title("World Space")
ax3.set_title("Clip Space")

plt.show()
