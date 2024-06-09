import numpy as np
import barycentric

point = np.array([1.0, 2.0, 3.0], dtype=np.float32)
tetrahedron = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
], dtype=np.float32)
barycentric_coords = np.zeros(4, dtype=np.float32)

barycentric.compute_barycentric(point, tetrahedron, barycentric_coords)

print("Barycentric Coordinates:", barycentric_coords)