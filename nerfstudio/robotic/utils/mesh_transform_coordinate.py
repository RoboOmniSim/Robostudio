import open3d as o3d
import numpy as np
import torch
import os


def transform_mesh(path):
    # Load the mesh

    ply_files = [f for f in os.listdir(path) if f.endswith('.obj')]
    point_clouds = []
    for file in ply_files:
        file_path = os.path.join(path, file)
        file_path_openglsave_name = file_path.split('.')[0] + '_opengl.obj'
        file_path_opencvsave_name = file_path.split('.')[0] + '_opencv.obj'
        mesh = o3d.io.read_triangle_mesh(file_path)
        mesh1 = o3d.io.read_triangle_mesh(file_path)
        vertices = np.asarray(mesh.vertices)

        point_cloud_opencv = np.zeros_like(vertices)
        point_cloud_opencv[:, 0] = -vertices[:, 0]
        point_cloud_opencv[:, 1] = -vertices[:, 1]
        point_cloud_opencv[:, 2] = vertices[:, 2]


        point_cloud_opengl = np.zeros_like(vertices)
        point_cloud_opengl[:, 0] = -vertices[:, 0]
        point_cloud_opengl[:, 1] = vertices[:, 1]
        point_cloud_opengl[:, 2] = -vertices[:, 2]
        
        mesh.vertices = o3d.utility.Vector3dVector(point_cloud_opengl)
        o3d.io.write_triangle_mesh(file_path_openglsave_name, mesh)
        mesh1.vertices = o3d.utility.Vector3dVector(point_cloud_opencv)
        o3d.io.write_triangle_mesh(file_path_opencvsave_name, mesh1)
    return point_clouds
    


if __name__ == '__main__':

    path=''
    transform_mesh(path)