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


def apply_transformation(gs_mesh_list_vertices,gs_mesh_list_faces,coordinate_info,num_linkes):
    urdf_mesh_vertices_list=[[]]*num_linkes
    urdf_mesh_faces_list=[[]]*num_linkes

    raw_mesh_vertices_list=[[]]*num_linkes
    raw_mesh_faces_list=[[]]*num_linkes
    R_raw2ori_list=[[]]*num_linkes   
    T_raw2ori_list=[[]]*num_linkes
    R_ori2raw_list=[[]]*num_linkes
    T_ori2raw_list=[[]]*num_linkes
    Matrix_reorientated_list=[[]]*num_linkes
    for i in range(num_linkes):
        mesh=trimesh.Trimesh(vertices=gs_mesh_list_vertices[i],faces=gs_mesh_list_faces[i])
        raw_mesh=trimesh.Trimesh(vertices=mesh.vertices,faces=mesh.faces)


        raw_mesh_vertices_list[i]=raw_mesh.vertices
        raw_mesh_faces_list[i]=raw_mesh.faces
         
        if i==7:
            matrix=mesh.apply_obb()  # recenter
        else:
            matrix=mesh.apply_obb()  # recenter 
        # mesh.export(f'test_{i}.ply', file_type='ply')  #
        if usepointcloud:
            reshaped_vertices=mesh.vertices.reshape(3,-1)
            reshaped_raw_vertices=raw_mesh.vertices.reshape(3,-1)

            R_raw2ori,T_raw2ori=rigid_transform_3D(reshaped_raw_vertices, reshaped_vertices)

            R_ori2raw,T_ori2raw=rigid_transform_3D(reshaped_vertices, reshaped_raw_vertices)
        else:
            R_raw2ori=0
            T_raw2ori=0
            R_ori2raw=0
            T_ori2raw=0
        R_raw2ori_list[i]=R_raw2ori
        T_raw2ori_list[i]=T_raw2ori
        R_ori2raw_list[i]=R_ori2raw
        T_ori2raw_list[i]=T_ori2raw
        Matrix_reorientated_list[i]=matrix
        # reset coordinate
        # mesh.apply_transform(coordinate_info[i])

        # reset center by moving the center to the motor engine location 
        # mesh.apply_transform(new_center_list[i])
        urdf_mesh_vertices_list[i]=mesh.vertices
        urdf_mesh_faces_list[i]=mesh.faces
    return urdf_mesh_vertices_list,urdf_mesh_faces_list,raw_mesh_vertices_list,raw_mesh_faces_list,R_raw2ori_list,T_raw2ori_list,R_ori2raw_list,T_ori2raw_list,Matrix_reorientated_list



def remap_basedon_kinematic(vertices, recenter_matrix):
    """
    Recenter the vertices based on the kinematic information.
    """
    # Convert the vertices to homogeneous coordinates

    recenter_matrix=np.array(recenter_matrix)
    # Apply the recenter matrix to the vertices
    rotation_rela=recenter_matrix[:3,:3]

    translation_rela=recenter_matrix[:3,3]        

    forward_point=  np.array(vertices @ rotation_rela.T+ translation_rela )

    return forward_point






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

def compute_center(bbox,num_linkes):
    center_list=[[]]*num_linkes # 7 is the number of linkes


    # this is the porpotion of the engine center against its bbox
    center_list[0]=np.array([0,0,0])
    center_list[1]=np.array([0,0,0])
    center_list[2]=np.array([0,0,0])
    center_list[3]=np.array([0,0,0])
    center_list[4]=np.array([0,0,0])
    center_list[5]=np.array([0,0,0])
    center_list[6]=np.array([0,0,0])

    
    for i in range(num_linkes):
        upper_box = bbox[i][1] 
        lower_box = bbox[i][0]
        proportion=upper_box/(np.abs(upper_box)+np.abs(lower_box))
        proportion=np.maximum(proportion,0.005)
        center_list[i]=proportion

    return center_list

def computer_center_move(bbox,num_linkes):
    center_move_list=[[]]*num_linkes

    center_move_list[0]=np.array([0,0,0])
    center_move_list[1]=np.array([0,0,0])
    center_move_list[2]=np.array([0,0,0])
    center_move_list[3]=np.array([0,0,0])
    center_move_list[4]=np.array([0,0,0])
    center_move_list[5]=np.array([0,0,0])

    center_move_list[6]=np.array([0,0,0]) # if all 7 links
    
    for i in range(num_linkes):
        # # manual for non-centric motor engine ,sugar case
        if bbox[i][0][2]>0:
            move_rate_x=(abs(bbox[i][0][0])+abs(bbox[i][1][1]))/2
            move_rate_y=(abs(bbox[i][0][1])+abs(bbox[i][1][1]))/2
            move_rate_z=(abs(bbox[i][0][2])+abs(bbox[i][1][2]))/2
            center_move_list[i]=np.array([move_rate_x,move_rate_y,move_rate_z])

        half_center=(abs(bbox[i][0][2])+abs(bbox[i][1][2]))/2

        
        if bbox[i][0][2]<0 :
            if (half_center*0.8)>abs(bbox[i][0][2]) :
                move_rate=half_center-abs(bbox[i][0][2]) 
                center_move_list[i]=np.array([0,0,-move_rate])



        
    return center_move_list

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

def edit_with_transformation(urdf_mesh_vertices_list,new_center_list):
    for i in range(len(urdf_mesh_vertices_list)):
        vertices=urdf_mesh_vertices_list[i]+new_center_list[i]
        
        urdf_mesh_vertices_list[i]=vertices
    return urdf_mesh_vertices_list


def apply_cordinate_shift_gt(urdf_mesh_vertices_list,coordinate_info_R,num_linkes):
    urdf_mesh_vertices_list_reori=[[]]*num_linkes
    for i in range(num_linkes):
        re=np.dot(urdf_mesh_vertices_list[i],coordinate_info_R[i])
        
        urdf_mesh_vertices_list_reori[i]=re
    return urdf_mesh_vertices_list_reori

def compute_bbox_gs(vertices):
     
    bbox = np.array([np.min(vertices, axis=0), np.max(vertices, axis=0)])
    return bbox
