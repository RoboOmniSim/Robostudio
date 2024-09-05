
from plyfile import PlyData, PlyElement
import numpy as np
import torch
import open3d as o3d
import trimesh
import os

from nerfstudio.robotic.kinematic.uniform_kinematic import *
from nerfstudio.robotic.export_util.urdf_utils.urdf_helper import find_lower_plane_center


def convert_pointcloud_to_ply(path):
    vertices_save_list=[]
    faces_save_list=[]
    file_name_list=[]
    bbox_save_list=[]
    bbox_reoriented_save_list=[]
    corners_list=[]
     # vertices, faces
    path_list=os.listdir(path)
    path_list.sort()

    for file in path_list:
        if file.endswith('.ply'):
            # obj_path = os.path.join(path, file)
            ply_path = os.path.join(path, file)
            

            pcd = o3d.io.read_point_cloud(ply_path)
            vertices = np.asarray(pcd.points)
            pcd.estimate_normals()

            # estimate radius for rolling ball
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 1.5 * avg_dist   

            raw_mesh, _ = pcd.compute_convex_hull()

            # create the triangular mesh with the vertices and faces from open3d
            tri_mesh = trimesh.Trimesh(np.asarray(raw_mesh.vertices), np.asarray(raw_mesh.triangles),
                                    vertex_normals=np.asarray(raw_mesh.vertex_normals))
            
            
            bbox = tri_mesh.bounding_box.bounds
            
            bbox_save_list.append(bbox)
            bbox_reorient = tri_mesh.bounding_box_oriented.bounds
            corners=trimesh.bounds.corners(bbox_reorient)
            corners_list.append(corners)
            bbox_reoriented_save_list.append(bbox_reorient)
            # vertices = tri_mesh.vertices
            faces = tri_mesh.faces
            # Export the mesh as a PLY file
            # mesh.export(ply_path, file_type='ply')
            
            vertices_save_list.append(vertices)
            faces_save_list.append(faces)
            file_name_list.append(file)
    return vertices_save_list,faces_save_list,file_name_list,bbox_save_list,bbox_reoriented_save_list,corners_list


import argparse

if __name__=="__main__":
    # full_bbox_path='/home/lou/gs/nerfstudio/exports/splat/no_downscale/gripper_grasp_open/full_part' ## gripper close part
    parser = argparse.ArgumentParser(description="export bbox list information from ply file")
    
    # Add arguments
    parser.add_argument('--full_bbox_path', type=str, help='path to the full part ply file')
    parser.add_argument('--save_path', type=str, help='path to save bbox list information. varies based on experiment type')
    # parser.add_argument('--flag', action='store_true', help='A boolean flag')
    
    full_bbox_path=parser.parse_args().full_bbox_path

    vertices_save_list,faces_save_list,file_name_list,bbox_save_list,bbox_reoriented_save_list,corners_list=convert_pointcloud_to_ply(full_bbox_path)

    

    recenter_vector=find_lower_plane_center(corners_list[0])
    save_path=parser.parse_args().save_path
    np.savetxt(os.path.join(save_path,"bbox_list.txt"), np.array(bbox_save_list).reshape(-1, np.array(bbox_save_list).shape[-1]))


    print("recenter_vector",recenter_vector)