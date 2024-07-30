import torch
import numpy as np
import pypose as pp
import os
import cv2
import open3d as o3d
import trimesh



import argparse
from nerfstudio.robotic.kinematic.uniform_kinematic import *
from nerfstudio.robotic.export_util.urdf_utils.urdf_config import *

from nerfstudio.robotic.kinematic.gripper_utils import reflect_x_axis,reflect_y_axis,reflect_z_axis

from nerfstudio.robotic.kinematic.control_helper import *

def recenter_basedon_kinematic(vertices, recenter_matrix):
    """
    Recenter the vertices based on the kinematic information.
    """
    # Convert the vertices to homogeneous coordinates
    vertices = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
    recenter_matrix=np.array(recenter_matrix)
    # Apply the recenter matrix to the vertices
    vertices = vertices @ recenter_matrix.T
    return vertices[:, :3]












def convert_obj_to_ply(path,num_linkes):

    points_list=[[]]*num_linkes
    file_name_list=[[]]*num_linkes
    face_list=[[]]*num_linkes
    color_list=[[]]*num_linkes
    normals_list=[[]]*num_linkes
    bbox_save_list=[]
     # vertices, faces
    path_list=os.listdir(path)
    path_list.sort()
    index=0
    for file in path_list:
        if file.endswith('.obj'):
            obj_path = os.path.join(path, file)

            obj_data = o3d.io.read_triangle_mesh(obj_path)


            triangle_clusters, cluster_n_triangles, cluster_area = obj_data.cluster_connected_triangles()
            

            largest_cluster_idx = np.argmax(cluster_area)
            triangles_to_remove = [i for i, c in enumerate(triangle_clusters) if c != largest_cluster_idx]
            

            obj_data.remove_triangles_by_index(triangles_to_remove)

            obj_data.remove_unreferenced_vertices()

            vertices=np.asarray(obj_data.vertices)
            face=np.asarray(obj_data.triangles)
            color=np.asarray(obj_data.vertex_colors)
            normals=np.asarray(obj_data.vertex_normals)
            # Load the OBJ file

            tri_mesh = trimesh.Trimesh(np.asarray(vertices), np.asarray(face),
                                    vertex_normals=np.asarray(normals))
            
            
            bbox = tri_mesh.bounding_box.bounds
            bbox_save_list.append(bbox)

            points_list[index]=vertices
            face_list[index]=face
            color_list[index]=color
            normals_list[index]=normals

            
            file_name_list[index]=file
            index+=1
    return points_list,face_list,color_list,normals_list,file_name_list,bbox_save_list


def convert_obj_to_ply_scene(path):
    scene_list=[]
    file_name_list_scene=[]
    path_list=os.listdir(path)
    path_list.sort()
    for file in path_list:
        if file.endswith('.obj'):
            obj_path = os.path.join(path, file)

            

            # Load the OBJ file
            mesh = trimesh.load(obj_path)

            scene_list.append(mesh)
        file_name_list_scene.append(file)
    return scene_list,file_name_list_scene

def get_8_corners_from_bbox(bbox,scale_factor):
    (xmin, ymin, zmin), (xmax, ymax, zmax) = bbox

    # time scale_factor to each corner
    xmin, ymin, zmin = xmin / scale_factor[0], ymin / scale_factor[1], zmin / scale_factor[2]
    xmax, ymax, zmax = xmax / scale_factor[0], ymax / scale_factor[1], zmax / scale_factor[2]
    
    corners = np.array([
        [xmin, ymin, zmin],
        [xmin, ymin, zmax],
        [xmin, ymax, zmin],
        [xmin, ymax, zmax],
        [xmax, ymin, zmin],
        [xmax, ymin, zmax],
        [xmax, ymax, zmin],
        [xmax, ymax, zmax],
    ])

    return corners


def get_bbox_from_8_corners(corners):
    min_corner = np.min(corners, axis=0)
    max_corner = np.max(corners, axis=0)

    bbox = (min_corner.tolist(), max_corner.tolist())
    return bbox



def save_obj(urdf_mesh_vertices_list,urdf_mesh_faces_list,color_list,normals_list,save_path,num_linkes,filename_list_used):
    for i in range(num_linkes):
        deform_point=urdf_mesh_vertices_list[i]
        select_faces=urdf_mesh_faces_list[i]
        select_color=color_list[i]
        select_normals=normals_list[i]
        file_name=filename_list_used[i]
          
        output_mesh=o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(deform_point),triangles=o3d.utility.Vector3iVector(select_faces))
        output_mesh.vertex_colors=o3d.utility.Vector3dVector(select_color)
        from copy import deepcopy
        select_normals_copy=deepcopy(select_normals)
        output_mesh.vertex_normals=o3d.utility.Vector3dVector(select_normals_copy)

                
        # o3d.io.write_point_cloud(f'/home/lou/gs/nerfstudio/exports/splat/no_downscale/group1_bbox_fix/forward/deform_point{i}_deform.ply', o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(deform_point)))
        o3d.io.write_triangle_mesh(os.path.join(save_path,file_name),output_mesh)


# Function to calculate all transformation matrices and the final matrix
def calculate_transformations_mdh_urdf(initial_position,joint_angles_degrees, a, alpha, d,i=0,add_gripper=False,flip_x_coordinate=False,flip_y_coordinate=False,flip_z_coordinate=False):
    """
    calulate transformation matrices based dh parameters

    Args:
    i: timestamp
    joint_angles_degrees: list of joint angles in degrees
    a: list of a parameters
    alpha: list of alpha parameters
    d: list of d parameters

    
    
    
    """
    state_position=initial_position


    joint_angles_radians_raw = joint_angles_degrees
    joint_angle_deform= np.array(state_position)
    joint_angle_deform=np.round(joint_angle_deform, 3)



    joint_angles_radians=joint_angles_radians_raw+joint_angle_deform


    transformations = []

    j=0
    gripper_index_list=[7,9]
    
    for theta, a_i, alpha_i, d_i in zip(joint_angles_radians, a, alpha, d):
        

        # apply edit gripper mdh for the control left down gripper and right down gripper
        if j==9:
            T_temp=create_transformation_matrix_mdh_gripper(theta, a_i, alpha_i, d_i)
        
        elif j==7:
            T_temp=create_transformation_matrix_mdh_gripper(theta, a_i, alpha_i, d_i)
        else:
            T_temp=create_transformation_matrix_mdh(theta, a_i, alpha_i, d_i)

        # gripper has different base coordinate so need new flip coordinate
        if flip_x_coordinate:
            if j not in gripper_index_list:
                T_temp=reflect_x_axis(T_temp) # for different base coordinate, the default base coordinate is - + - 
            else:
                # T_temp=create_transformation_matrix_mdh_gripper_reflect_x_coordinate(theta, a_i, alpha_i, d_i) # for different base coordinate, the default base coordinate is - + - since mdh gripper is different from mdh
                T_temp=reflect_x_axis_only(T_temp)
        if flip_y_coordinate:
            if j not in gripper_index_list:
                T_temp=reflect_y_axis(T_temp) # for different base coordinate, the default base coordinate is - + - 
            else:
                # T_temp=create_transformation_matrix_mdh_gripper_reflect_x_coordinate(theta, a_i, alpha_i, d_i) # for different base coordinate, the default base coordinate is - + - since mdh gripper is different from mdh
                T_temp=reflect_y_axis_only(T_temp)
        if flip_z_coordinate:
            if j not in gripper_index_list:
                T_temp=reflect_z_axis(T_temp)
            else:
                T_temp=reflect_z_axis_only(T_temp)

        # for gripper right 10,11, it is connect to 7 
        transformations.append(T_temp)

        j+=1
    # Calculate the final transformation from the base to the end-effector
    final_transformation = np.eye(4)
    final_transformations_list=[[]]*len(joint_angles_radians)

    p=0
    if add_gripper:

        
        for transformation in transformations:
            if p==9:
                gripper_move= final_transformations_list[6]
                final_transformation = np.dot(gripper_move, transformation)
                final_transformations_list[p]=final_transformation
            elif p==10:
                gripper_right_move= final_transformations_list[9]
                final_transformation = np.dot(gripper_right_move, transformation)
                final_transformations_list[p]=final_transformation
            else:
                final_transformation = np.dot(final_transformation, transformation)
                final_transformations_list[p]=final_transformation
            p+=1
    else:
        for transformation in transformations:
            final_transformation = np.dot(final_transformation, transformation)
            final_transformations_list[p]=final_transformation
            p+=1
    return transformations, final_transformations_list


# run command: python nerfstudio/robotic/export_util/urdf_utils/urdf.py --part_path ./dataset/roboarm2/urdf/2dgs/arm --save_path ./dataset/roboarm2/urdf/2dgs/recenter_mesh --kinematic_info_path ./dataset/roboarm2/urdf/2dgs/kinematic/kinematic_info.yaml --experiment_type cr3 --scale_factor_gt 1.0 --num_links 7


def main():
    parser = argparse.ArgumentParser(description="export bbox list information from ply file")
    
    # Add arguments
    parser.add_argument('--part_path', type=str, help='path to the full part mesh file')
    parser.add_argument('--save_path', type=str, help='path to save bbox list information and recentered urdf. varies based on experiment type')
    parser.add_argument('--kinematic_info_path', type=str, help='path to load scale and kinematic information. varies based on robotic arm')
    parser.add_argument('--experiment_type', type=str, help='type of the experiment')
    parser.add_argument('--scale_factor_gt', type=float, help='scale factor of the part')
    parser.add_argument('--num_links', type=int, help='number of links in the robotic arm')
    parser.add_argument('--base_gt', type=float, help='scale factor of the base')

    # scale_factor
    args = parser.parse_args()

    if os.path.exists(args.save_path)==False:
        os.makedirs(args.save_path)

    points_list,face_list,color_list,normals_list,file_name_list,bbox_save_list=convert_obj_to_ply(args.part_path,args.num_links)
    # Load the part mesh
    bounding_box_corners = bbox_save_list[0]
    
    # this method works for  - + - coordinate system for recenter_vector
    extent_base=bounding_box_corners[1]-bounding_box_corners[0]

    # compare with the base_gt and bounding boxe base

    # test this scale factor 
    scale_factor_base=extent_base/args.base_gt
    scale_factor=scale_factor_base
    # Get the coordinates of the lower plane center
    center_vector_compute = bounding_box_corners.mean(axis=0)
    center_vector_compute[2] = bounding_box_corners[0][2]
    center_matrix=np.eye(4)
    center_matrix[:3,3]=center_vector_compute*-1
    # Load the kinematic information


    Urdfinfo=Urdfconfig()
    Urdfinfo.setup_params(args.kinematic_info_path)

    a,alpha,d,joint_angles_degrees=Urdfinfo.a,Urdfinfo.alpha,Urdfinfo.d,Urdfinfo.joint_angles_degrees


    scale_a=np.array([1,1,1,1,1,1])/scale_factor[2]  # s1 is z axis for a1, s2 is z axis for a2=0, s3 is z axis for a3,  s4 is z axis for a4
    scale_d=np.array([1,1,1,1,1,1])/scale_factor[2] #  s1 is x axis for d1 ,s4 and s6  is y axis 



    scale_d[3]=1/scale_factor[1]
    scale_d[5]=1/scale_factor[1]

    a=a*scale_a
    d=d*scale_d
    initial_position=np.zeros_like(joint_angles_degrees)
    # apply center_vector_compute to all points
    for i in range(args.num_links):
        points_list[i] = recenter_basedon_kinematic(points_list[i], center_matrix)


    transformations, final_transformations_list=calculate_transformations_mdh_urdf(initial_position,joint_angles_degrees, a, alpha, d,i=0,add_gripper=Urdfinfo.add_gripper,flip_x_coordinate=Urdfinfo.flip_x_coordinate,flip_y_coordinate=Urdfinfo.flip_y_coordinate,flip_z_coordinate=Urdfinfo.flip_z_coordinate)
    # recenter of each part based on edited kinematic
    remap_matrix=inverse_affine_transformation(final_transformations_list)

    # add a identity matrix for the base
    remap_matrix.insert(0,np.eye(4))
    # no gripper and camera this version
    for i in range(args.num_links):
        points_list[i] = recenter_basedon_kinematic(points_list[i], remap_matrix[i])


    save_obj(points_list,face_list,color_list,normals_list,args.save_path,args.num_links,file_name_list)


if __name__=="__main__":
    main()