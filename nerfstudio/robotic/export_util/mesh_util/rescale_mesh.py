# we build a cage with tetrahedra based on robotic arm model

# and rescale this cage to new scenes with different scales follow barycentric coordinates


import numpy as np
import open3d as o3d
import os
import trimesh
import barycentric

# point = np.array([1.0, 2.0, 3.0], dtype=np.float32)
# tetrahedron = np.array([
#     [0.0, 0.0, 0.0],
#     [1.0, 0.0, 0.0],
#     [0.0, 1.0, 0.0],
#     [0.0, 0.0, 1.0]
# ], dtype=np.float32)
# barycentric_coords = np.zeros(4, dtype=np.float32)

# barycentric.compute_barycentric(point, tetrahedron, barycentric_coords)

# print("Barycentric Coordinates:", barycentric_coords)

def get_barycentric_coordinates(mesh, points):
    barycentric_coords = np.zeros((len(points), 4), dtype=np.float32)
    for i, point in enumerate(points):
        barycentric.compute_barycentric(point, mesh.vertices, barycentric_coords[i])
    return barycentric_coords


def rescale_mesh(mesh, barycentric_coords, scale):
    new_vertices = mesh.vertices * scale
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    new_mesh.triangles = mesh.triangles
    return new_mesh
