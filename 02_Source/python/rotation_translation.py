import numpy as np
import pyplot from matplotlib

num_pts = 100

# u, v and w are IMU and Camera coordinates - these rotate together with the camera head.
uvw = np.zeros((3,num_pts))


# rotation quaternion resembling the orientation of u, v and w regarding x, y, z
p = np.zeros((4,num_pts))

# x, y and z are the spatial coordinates
xyz = np.zeros((3,num_pts))